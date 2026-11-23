"""The sandbox that the sandbox tools of one chat session share during a run."""

import hashlib
import threading
from collections.abc import Mapping, Sequence
from uuid import UUID

from onyx.db.engine.sql_engine import get_session_with_tenant
from onyx.db.models import User
from onyx.db.sandbox import get_or_create_sandbox
from onyx.db.skill import list_chat_skills
from onyx.server.features.build.sandbox.harness_base import HarnessSandbox
from onyx.server.features.build.sandbox.models import FileSet, HarnessSandboxRef
from onyx.skills.models import ChatSkill
from onyx.tools.tool_implementations.sandbox.file_copies import TurnFileCopies
from onyx.tools.tool_implementations.sandbox.models import VmChatFile
from onyx.utils.logger import setup_logger

logger = setup_logger()


class SandboxTurn:
    """The user's sandbox, seen from one chat session, for the tools built
    together for one model response. The runs of the response (the root run
    and its subagent runs) share it.

    The first call resolves the user's Sandbox row, so a resolved row means
    that a call ran. Each call renews the sandbox's turn mark before it runs,
    which keeps the sandbox busy. After the first call, each release of the
    root run (it finishes or suspends, and its work and the work of its
    foreground children drain) ends the turn: `end` is the root agent's
    on_drained hook (`chat/execution.py:create_chat_agent`). Subagent runs never end it,
    since the root can be mid-call when they release; a background subagent
    that outlives the root leaves its renewed mark to expire.

    `vm_chat_files` are the chat's files in the VM. In a turn that copies
    them (`copy_vm_chat_files`), the first call starts `file_copies`, their
    background copy; the root chat agent adds its status line to sandbox and
    code mode results (`chat/agent.py`).
    """

    def __init__(
        self,
        *,
        tenant_id: str,
        user: User,
        chat_session_id: UUID,
        vm_chat_files: Sequence[VmChatFile],
        copy_vm_chat_files: bool,
    ) -> None:
        self._tenant_id = tenant_id
        self._user = user
        self._chat_session_id = chat_session_id
        self.vm_chat_files = vm_chat_files
        self._copy_vm_chat_files = copy_vm_chat_files
        self._lock = threading.Lock()
        self._ref: HarnessSandboxRef | None = None
        self._skills_lock = threading.Lock()
        self.file_copies: TurnFileCopies | None = None
        self._reads: dict[str, int] = {}
        self._wanted: set[str] = set()
        self._resolved: set[str] = set()
        self._skills: dict[str, ChatSkill] = {}
        self._skill_files: dict[str, FileSet] = {}

    def want_skills(self, reads: Mapping[str, int]) -> None:
        """Copy these skills (name to read count) before the next call. A
        higher count re-resolves the skill's version."""
        with self._lock:
            self._resolved -= {
                name
                for name, count in reads.items()
                if count > self._reads.get(name, 0)
            }
            self._reads = dict(reads)
            self._wanted.update(reads)

    @property
    def sandbox(self) -> HarnessSandbox:
        # Deferred: the sandbox client stack loads only when a tool runs.
        from onyx.server.features.build.sandbox.harness_factory import (
            get_harness_sandbox,
        )

        return get_harness_sandbox()

    def start_call(self) -> HarnessSandboxRef:
        """The sandbox reference for one call, with the turn mark renewed.
        The first call starts the copy of the chat's files."""
        with self._lock:
            ref = self._ref
        first = False
        if ref is None:
            # Outside the lock: concurrent first calls get the same row.
            with get_session_with_tenant(tenant_id=self._tenant_id) as db_session:
                sandbox_id = get_or_create_sandbox(db_session, self._user.id).id
            with self._lock:
                if self._ref is None:
                    self._ref = HarnessSandboxRef(
                        tenant_id=self._tenant_id,
                        sandbox_id=sandbox_id,
                        session_id=self._chat_session_id,
                    )
                    first = True
                ref = self._ref
        # The mark expires on its own, so a failed renewal only lets a
        # turn-end save run sooner; the call still runs.
        try:
            self.sandbox.begin_turn(ref)
        except Exception:
            logger.exception(
                "sandbox_turn_renew_failed sandbox=%s session=%s",
                ref.sandbox_id,
                ref.session_id,
            )
        if first and self._copy_vm_chat_files:
            # A copy that fails to start leaves the files to the next turn.
            try:
                self.file_copies = TurnFileCopies(ref, self.sandbox, self.vm_chat_files)
            except Exception:
                logger.exception(
                    "sandbox_file_copies_start_failed sandbox=%s session=%s",
                    ref.sandbox_id,
                    ref.session_id,
                )
        with self._skills_lock:
            try:
                self._install_skills(ref)
            except Exception:
                logger.exception(
                    "sandbox_skills_install_failed sandbox=%s session=%s",
                    ref.sandbox_id,
                    ref.session_id,
                )
        return ref

    def _install_skills(self, ref: HarnessSandboxRef) -> None:
        from onyx.skills.push import (
            build_chat_built_in_skill_files,
            build_chat_custom_skill_files,
        )

        with self._lock:
            unresolved = self._wanted - self._resolved
            reads = dict(self._reads)
        if unresolved:
            with get_session_with_tenant(tenant_id=self._tenant_id) as db_session:
                skills = list_chat_skills(db_session, self._user, built_ins=True)
            for name in unresolved:
                self._skills.pop(name, None)
                self._skill_files.pop(name, None)
                skill = next((s for s in skills if s.name == name), None)
                if skill is not None:
                    self._skills[name] = skill
            with self._lock:
                self._resolved |= {
                    name
                    for name in unresolved
                    if self._reads.get(name, 0) == reads.get(name, 0)
                }
        if not self._skills:
            return

        def load(name: str) -> FileSet | None:
            files = self._skill_files.get(name)
            if files is None:
                skill = self._skills[name]
                if skill.built_in_skill_id is not None:
                    files = build_chat_built_in_skill_files(skill)
                else:
                    with get_session_with_tenant(
                        tenant_id=self._tenant_id
                    ) as db_session:
                        files = build_chat_custom_skill_files(
                            skill, self._user, db_session
                        )
                if files is not None:
                    self._skill_files[name] = files
            return files

        def version(name: str, skill: ChatSkill) -> str:
            if skill.built_in_skill_id is None:
                return f"{skill.version_id}:{skill.path}"
            digest = hashlib.sha256()
            for path, data in sorted((load(name) or {}).items()):
                digest.update(f"{path}\0{len(data)}\0".encode())
                digest.update(data)
            return digest.hexdigest()

        self.sandbox.install_skills(
            ref,
            {name: version(name, skill) for name, skill in self._skills.items()},
            load,
        )

    def end(self) -> None:
        """End the turn if a call started one. Never raises."""
        with self._lock:
            ref = self._ref
        if ref is None:
            return
        # The turn mark expires on its own, so a failure only delays the
        # turn-end save.
        try:
            self.sandbox.end_turn(ref)
        except Exception:
            logger.exception(
                "sandbox_turn_end_failed sandbox=%s session=%s",
                ref.sandbox_id,
                ref.session_id,
            )
