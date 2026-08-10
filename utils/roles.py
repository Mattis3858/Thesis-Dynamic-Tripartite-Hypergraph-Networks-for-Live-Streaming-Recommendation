"""
Detect which tripartite column actually holds the streamers.

The KuaiLive preprocessing chain has two incompatible etype conventions (see
``remap_kuailive_etypes.py``). When the raw edge table is fed to the converter without the
remap, the converter labels ROOMS as streamers and vice versa, so in
``processed_data/<ds>/ml_<ds>.csv`` the column named ``streamer`` can in fact hold rooms and
``room`` can hold streamers. ``TripartiteData.streamer_node_ids`` / ``item_node_ids`` inherit
that naming, so any analysis that trusts the names silently measures the wrong entity.

Rather than hard-coding a swap (which would break again on a correctly remapped dataset), we
detect the roles from the data: a live room belongs to exactly one streamer, so the column
whose values each map to a single value of the other column is the ROOM, and the other is the
STREAMER.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

ATTR_STREAMER_COL = "streamer_node_ids"  # the column literally named "streamer"
ATTR_ROOM_COL = "item_node_ids"  # the column literally named "room"


@dataclass
class RoleMapping:
    """Which data attribute holds the true streamers / rooms."""

    streamer_attr: str
    room_attr: str
    num_streamers: int
    num_rooms: int
    swapped: bool
    reason: str

    def describe(self) -> str:
        state = "SWAPPED (column names do not match the entities)" if self.swapped else "as named"
        return (
            f"Role detection: {state}; true streamers = data.{self.streamer_attr} "
            f"({self.num_streamers} distinct), rooms/items = data.{self.room_attr} "
            f"({self.num_rooms} distinct). {self.reason}"
        )


def _max_partners(a: np.ndarray, b: np.ndarray) -> int:
    """Largest number of distinct b-values attached to any single a-value."""
    order = np.argsort(a, kind="stable")
    a_sorted, b_sorted = a[order], b[order]
    bounds = np.flatnonzero(np.diff(a_sorted)) + 1
    return max((len(np.unique(chunk)) for chunk in np.split(b_sorted, bounds)), default=0)


def detect_roles(data, *, verbose: bool = True) -> RoleMapping:
    """
    Infer the true streamer/room columns of a TripartiteData split.

    Pass the FULL data (or train split) -- the nesting test needs enough coverage.
    """
    col_streamer = np.asarray(getattr(data, ATTR_STREAMER_COL), dtype=np.int64)
    col_room = np.asarray(getattr(data, ATTR_ROOM_COL), dtype=np.int64)

    n_col_streamer = len(np.unique(col_streamer))
    n_col_room = len(np.unique(col_room))

    streamer_to_room = _max_partners(col_streamer, col_room)
    room_to_streamer = _max_partners(col_room, col_streamer)

    if streamer_to_room == 1 and room_to_streamer > 1:
        # every "streamer"-column value sits under one "room"-column value => it IS a room
        mapping = RoleMapping(
            streamer_attr=ATTR_ROOM_COL, room_attr=ATTR_STREAMER_COL,
            num_streamers=n_col_room, num_rooms=n_col_streamer, swapped=True,
            reason="Each value of the 'streamer' column nests under a single 'room' value.",
        )
    elif room_to_streamer == 1 and streamer_to_room > 1:
        mapping = RoleMapping(
            streamer_attr=ATTR_STREAMER_COL, room_attr=ATTR_ROOM_COL,
            num_streamers=n_col_streamer, num_rooms=n_col_room, swapped=False,
            reason="Each value of the 'room' column nests under a single 'streamer' value.",
        )
    else:
        # No clean nesting (or both are 1:1). Fall back to cardinality: rooms outnumber streamers.
        swapped = n_col_room < n_col_streamer
        mapping = RoleMapping(
            streamer_attr=ATTR_ROOM_COL if swapped else ATTR_STREAMER_COL,
            room_attr=ATTR_STREAMER_COL if swapped else ATTR_ROOM_COL,
            num_streamers=min(n_col_streamer, n_col_room),
            num_rooms=max(n_col_streamer, n_col_room),
            swapped=swapped,
            reason=(
                "!! No clean room->streamer nesting found "
                f"(max partners: streamer-col->{streamer_to_room}, room-col->{room_to_streamer}); "
                "fell back to cardinality (fewer distinct values = streamers). VERIFY THIS."
            ),
        )

    if verbose:
        print(mapping.describe())
    return mapping


def role_ids(data, mapping: RoleMapping, role: str) -> np.ndarray:
    """Fetch the id array for 'user' | 'streamer' | 'room' under a detected mapping."""
    if role == "user":
        return np.asarray(data.user_node_ids, dtype=np.int64)
    if role == "streamer":
        return np.asarray(getattr(data, mapping.streamer_attr), dtype=np.int64)
    if role in ("room", "item"):
        return np.asarray(getattr(data, mapping.room_attr), dtype=np.int64)
    raise ValueError(f"Unknown role: {role}")
