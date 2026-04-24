"""
demo.py — 3D multi-floor warehouse scenario
"""
import asyncio, sys, os, math, random
sys.path.insert(0, os.path.dirname(__file__))
import robot_memory as rm

def make_emb(seeds, dim=128):
    rng = random.Random(sum(int(v*1000) for v in seeds))
    vec = [rng.gauss(s, 0.3) for s in (seeds*(dim//len(seeds)+1))[:dim]]
    n   = math.sqrt(sum(v*v for v in vec)) or 1.0
    return [v/n for v in vec]

# Multi-floor warehouse:  floor 0 = ground, floor 1 = mezzanine
ENTITIES = [
    # name, summary, type, x, y, z, floor, facing, pitch, bbox(dx,dy,dz), tags, words, emb_seed
    ("Entry Door",   "Main entry",       "place",   0.0,  0.0, 0.0, 0, 0.0,  0.0, (0.1,2.0,2.5), ["entrance"], ["door","entry"],          [0.8,0.1,0.1]),
    ("Shelf A1",     "Rack 1 — boxes",   "object",  5.0,  2.0, 0.0, 0, 90.0, 0.0, (0.5,3.0,2.0), ["shelf"],    ["box","storage","aisle_a"],[0.1,0.1,0.9]),
    ("Shelf A2",     "Rack 2 — tools",   "object",  5.0,  6.0, 0.0, 0, 90.0, 0.0, (0.5,3.0,2.0), ["shelf"],    ["tools","storage"],        [0.15,0.1,0.85]),
    ("Charging Bay", "Charging dock",    "place",   0.0, 10.0, 0.0, 0, 180.0,0.0, (1.5,1.5,1.0), ["power"],    ["charge","battery","dock"],[0.5,0.5,0.1]),
    ("Forklift",     "Electric forklift","vehicle", 8.0,  4.0, 0.0, 0, 270.0,0.0, (1.5,3.0,2.5), ["vehicle"],  ["lift","heavy","caution"], [0.1,0.9,0.1]),
    ("Camera 3",     "Ceiling camera",   "sensor",  4.0,  4.0, 4.5, 0, 225.0,-45.0,(0.1,0.1,0.1),["camera"],  ["vision","security"],      [0.1,0.5,0.5]),
    ("Exit B",       "Emergency exit",   "place",  10.0,  0.0, 0.0, 0, 90.0, 0.0, (0.1,2.0,2.5), ["exit"],    ["emergency","door","exit"],[0.8,0.1,0.1]),
    # floor 1 — mezzanine (z≈3.5m)
    ("Office",       "Manager office",   "place",   2.0,  2.0, 3.5, 1, 180.0,0.0, (4.0,4.0,2.5), ["office"],  ["desk","manager","admin"], [0.3,0.6,0.2]),
    ("Parts Store",  "Spare parts room", "object",  6.0,  8.0, 3.5, 1, 270.0,0.0, (3.0,3.0,2.5), ["storage"], ["parts","spare","tools"],  [0.2,0.2,0.8]),
    ("Stairwell A",  "Stairs floor 0-1", "place",   1.0,  1.0, 1.75,0, 0.0,  45.0,(1.5,1.5,3.5), ["stairs"],  ["stairs","access","floor"],[0.4,0.4,0.4]),
]

# Robot path: ground floor then up stairwell to mezzanine
ROBOT_PATH = [
    # ground floor sweep
    (0.0,  0.0, 0.0, 0, 0.0,   0.0),
    (1.0,  0.0, 0.0, 0, 0.0,   0.0),
    (2.0,  0.0, 0.0, 0, 0.0,   0.0),
    (3.0,  0.0, 0.0, 0, 0.0,   0.0),
    (4.0,  1.0, 0.0, 0, 90.0,  0.0),
    (5.0,  2.0, 0.0, 0, 45.0,  0.0),   # Shelf A1
    (5.0,  4.0, 0.0, 0, 90.0,  0.0),
    (5.0,  6.0, 0.0, 0, 90.0,  0.0),   # Shelf A2
    (3.0,  8.0, 0.0, 0, 135.0, 0.0),
    (0.0, 10.0, 0.0, 0, 180.0, 0.0),   # Charging Bay
    # back toward stairwell
    (1.0,  2.0, 0.0, 0, 45.0,  0.0),
    # climb stairwell (z ramps up, pitch changes)
    (1.0,  1.5, 0.5, 0, 0.0,  20.0),
    (1.0,  1.2, 1.0, 0, 0.0,  30.0),
    (1.0,  1.0, 1.5, 0, 0.0,  30.0),
    (1.0,  0.8, 2.0, 0, 0.0,  30.0),
    (1.0,  0.6, 2.5, 0, 0.0,  30.0),
    (1.0,  0.4, 3.0, 0, 0.0,  20.0),
    (1.0,  0.2, 3.5, 1, 0.0,   0.0),   # arrived on floor 1
    # mezzanine
    (2.0,  2.0, 3.5, 1, 45.0,  0.0),   # Office
    (4.0,  5.0, 3.5, 1, 90.0,  0.0),
    (6.0,  8.0, 3.5, 1, 45.0,  0.0),   # Parts Store
]


def sep(title=""):
    w = 70
    if title: print(f"\n{'─'*5} {title} {'─'*(w-len(title)-7)}")
    else:     print("─" * w)


async def main():
    sep("INIT")
    await rm.init_pool()
    session = await rm.TemporalSession.start()

    sep("SEEDING 3D ENTITIES")
    entity_ids: dict[str, str] = {}
    for name, summary, etype, x, y, z, fl, facing, pitch, bbox, tags, kwords, eseed in ENTITIES:
        eid = await rm.upsert_entity(
            name=name, summary=summary, entity_type=etype,
            x=x, y=y, z=z, floor_level=fl,
            facing_deg=facing, pitch_deg=pitch,
            bbox_dx=bbox[0], bbox_dy=bbox[1], bbox_dz=bbox[2],
            tags=tags, weight=1.0,
            runtime_embedding=make_emb(eseed),
        )
        await rm.add_info_node(
            entity_id=eid, crucial_words=kwords, weight=1.0,
            full_data=f"{name}: {summary}. Pos=({x},{y},{z}) fl={fl} facing={facing}°",
            embedding=make_emb(eseed, dim=1536),
        )
        entity_ids[name] = eid
        print(f"  ✓ fl={fl}  {name:<20}  xyz=({x:.1f},{y:.1f},{z:.1f})  "
              f"hdg={facing:.0f}°  pitch={pitch:.0f}°")

    sep("SEEDING EDGES")
    for n1, n2, rtype, rname in [
        ("Entry Door",  "Camera 3",    "monitored_by", "Camera covers entry"),
        ("Shelf A1",    "Shelf A2",    "adjacent_to",  "Same aisle"),
        ("Forklift",    "Shelf A1",    "services",     "Forklift loads A1"),
        ("Charging Bay","Forklift",    "charges",      "Bay charges forklift"),
        ("Entry Door",  "Exit B",      "connects",     "Both exits"),
        ("Stairwell A", "Office",      "connects",     "Stairs to office"),
        ("Stairwell A", "Parts Store", "connects",     "Stairs to parts"),
        ("Office",      "Parts Store", "near",         "Same floor"),
    ]:
        await rm.upsert_edge(entity_ids[n1], entity_ids[n2], rel_type=rtype, rel_name=rname)
        print(f"  ✓ {n1} ─[{rtype}]→ {n2}")

    sep("3D PATH RECORDING (runtime, no DB writes)")
    for i, (x, y, z, fl, hdg, ptch) in enumerate(ROBOT_PATH):
        rt_id = await rm.record_position(
            x, y, z, floor_level=fl, heading_deg=hdg, pitch_deg=ptch,
            session_id=session.session_id,
        )
        for name, _, _, ex, ey, ez, efl, _, _, _, _, _, _ in ENTITIES:
            d3 = math.sqrt((x-ex)**2+(y-ey)**2+(z-ez)**2)
            if d3 < 1.5:
                log_id = await session.latest_path_log_id()
                await session.log(
                    entity_id=entity_ids[name], x=x, y=y, z=z,
                    floor_level=fl, path_log_ref=log_id, notes=f"d3={d3:.2f}m",
                )
    rt = rm.get_runtime_map()
    print(f"  Runtime map: {rt.node_count} nodes  {rt.edge_count} edges")
    print(f"  Floor 0 nodes: {len(rt.nodes_on_floor(0))}  "
          f"Floor 1 nodes: {len(rt.nodes_on_floor(1))}")

    sep("FLUSH PATH TO DB (manual)")
    from robot_memory.pathmap import flush_path_to_db
    n, e = await flush_path_to_db(session_id=None)
    print(f"  ✓ {n} path_nodes  {e} path_edges committed (with 3D distances)")

    sep("ANCHOR ENTITIES (manual)")
    for name, eid in entity_ids.items():
        try:
            await rm.anchor_entity_to_map(eid)
            print(f"  ✓ {name}")
        except ValueError as err:
            print(f"  ✗ {name}: {err}")

    sep("think()  floor=0  @  (5.0, 4.0, 0.0)  r=4m")
    ctx = await rm.think(robot_x=5.0, robot_y=4.0, robot_z=0.0, floor_level=0, radius_m=4.0)
    print(ctx.summary())

    sep("think()  floor=1  @  (3.0, 5.0, 3.5)  r=6m")
    ctx1 = await rm.think(robot_x=3.0, robot_y=5.0, robot_z=3.5, floor_level=1, radius_m=6.0)
    print(ctx1.summary())

    sep("think_nearest()  Shelf A1 → k=4  (all floors, r=15m)")
    nearest = await rm.think_nearest(entity_ids["Shelf A1"], k=4, radius_m=15.0)
    for ent, d3 in nearest:
        print(f"  {ent.name:<22}  d3={d3:.2f}m  floor={ent.floor_level}"
              f"  xyz=({ent.x:.1f},{ent.y:.1f},{ent.z:.1f})")

    sep("think_path()  Entry Door → Parts Store (cross-floor)")
    result = await rm.think_path(entity_ids["Entry Door"], entity_ids["Parts Store"])
    if result:
        print(f"  ✓ {len(result.waypoints)} waypoints")
        print(f"    dist_3d={result.total_dist_3d_m:.2f}m  "
              f"dist_2d={result.total_dist_2d_m:.2f}m  "
              f"ascent={result.total_ascent_m:.2f}m  "
              f"descent={result.total_descent_m:.2f}m")
        trans = result.floor_transitions()
        if trans:
            print(f"    floor transitions: {trans}")
        for wp in result.waypoints:
            print(f"    ({wp.x:.1f},{wp.y:.1f},{wp.z:.1f})  fl={wp.floor_level}")
    else:
        print("  ✗ No path found")

    sep("think_path()  Charging Bay → Office (cross-floor)")
    result2 = await rm.think_path(entity_ids["Charging Bay"], entity_ids["Office"])
    if result2:
        print(f"  ✓ {len(result2.waypoints)} waypoints  "
              f"3d={result2.total_dist_3d_m:.2f}m  "
              f"ascent={result2.total_ascent_m:.2f}m")
        for wp in result2.waypoints:
            print(f"    ({wp.x:.1f},{wp.y:.1f},{wp.z:.1f})  fl={wp.floor_level}")
    else:
        print("  ✗ No path found")

    sep("deep_think()")
    dctx = await rm.deep_think()
    print(dctx.summary())
    print("  Floor breakdown:")
    for fl in sorted(set(e.floor_level for e in dctx.all_entities)):
        ents = dctx.entities_on_floor(fl)
        print(f"    floor {fl}: {[e.name for e in ents]}")

    sep("TEMPORAL SUMMARY")
    await session.dump_summary()

    sep("DONE")
    await rm.close_pool()


if __name__ == "__main__":
    asyncio.run(main())