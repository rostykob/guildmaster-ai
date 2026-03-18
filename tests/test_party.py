"""Tests for Party domain model."""

from __future__ import annotations

from guildmaster_ai.core.party import Party, PartyMember


class TestParty:
    def test_creation(self) -> None:
        party = Party(name="Alpha Team", leader_id="leader-1")
        assert party.name == "Alpha Team"
        assert party.leader_id == "leader-1"
        assert party.members == []
        assert party.quest_id is None
        assert party.disbanded_at is None

    def test_add_member(self) -> None:
        party = Party(name="Alpha Team", leader_id="leader-1")
        party.add_member("adv-1")
        party.add_member("adv-2", role="scout")
        assert len(party.members) == 2
        assert party.members[0].adventurer_id == "adv-1"
        assert party.members[0].role == "member"
        assert party.members[1].role == "scout"

    def test_member_ids(self) -> None:
        party = Party(name="Alpha Team", leader_id="leader-1")
        party.add_member("adv-1")
        party.add_member("adv-2")
        assert party.member_ids == ["adv-1", "adv-2"]

    def test_disband(self) -> None:
        party = Party(name="Alpha Team", leader_id="leader-1")
        assert party.disbanded_at is None
        party.disband()
        assert party.disbanded_at is not None

    def test_quest_assignment(self) -> None:
        party = Party(name="Alpha Team", leader_id="leader-1", quest_id="quest-42")
        assert party.quest_id == "quest-42"

    def test_party_member_model(self) -> None:
        member = PartyMember(adventurer_id="adv-1", role="leader")
        assert member.adventurer_id == "adv-1"
        assert member.role == "leader"

    def test_auto_id(self) -> None:
        p1 = Party(name="A", leader_id="l")
        p2 = Party(name="B", leader_id="l")
        assert p1.id != p2.id
