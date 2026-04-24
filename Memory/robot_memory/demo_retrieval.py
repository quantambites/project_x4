"""
demo_retrieval.py — 3D retrieval scenarios
Four demos using a two-floor warehouse with full 3D positions.

  DEMO 1 — Path search cross-floor (ground → mezzanine)
  DEMO 2 — K-nearest 3D + per-floor filter + similarity
  DEMO 3 — Local window full-info (floor-scoped)
  DEMO 4 — Deep search + 3D-bounded vector search
"""
import asyncio, sys, os, math, random
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
import robot_memory as rm

def make_emb(seeds, dim=128):
    rng = random.Random(sum(int(v*1000) for v in seeds))
    vec = [rng.gauss(s,0.3) for s in (seeds*(dim//len(seeds)+1))[:dim]]
    n   = math.sqrt(sum(v*v for v in vec)) or 1.0
    return [v/n for v in vec]

# Semantic cluster query vectors
Q_STORAGE   = make_emb([0.1,0.1,0.9])
Q_EXIT      = make_emb([0.8,0.1,0.1])
Q_UPPER     = make_emb([0.3,0.6,0.2])

ENTITIES = [
    ("Entry Door",   "Main entry",        "place",   0.0,  0.0, 0.0, 0, 0.0,   0.0, ["door","entry"],          [0.8,0.1,0.1]),
    ("Shelf A1",     "Rack 1 — boxes",    "object",  5.0,  2.0, 0.0, 0, 90.0,  0.0, ["box","storage"],          [0.1,0.1,0.9]),
    ("Shelf A2",     "Rack 2 — tools",    "object",  5.0,  6.0, 0.0, 0, 90.0,  0.0, ["tools","storage"],        [0.15,0.1,0.85]),
    ("Charging Bay", "Charging dock",     "place",   0.0, 10.0, 0.0, 0, 180.0, 0.0, ["charge","dock"],          [0.5,0.5,0.1]),
    ("Forklift",     "Electric forklift", "vehicle", 8.0,  4.0, 0.0, 0, 270.0, 0.0, ["lift","heavy"],           [0.1,0.9,0.1]),
    ("Camera 3",     "Ceiling camera",    "sensor",  4.0,  4.0, 4.5, 0, 225.0,-45.0,["vision","security"],      [0.1,0.5,0.5]),
    ("Exit B",       "Emergency exit",    "place",  10.0,  0.0, 0.0, 0, 90.0,  0.0, ["emergency","exit"],       [0.8,0.1,0.1]),
    ("Office",       "Manager office",    "place",   2.0,  2.0, 3.5, 1, 180.0, 0.0, ["desk","admin"],           [0.3,0.6,0.2]),
    ("Parts Store",  "Spare parts room",  "object",  6.0,  8.0, 3.5, 1, 270.0, 0.0, ["parts","spare","tools"],  [0.2,0.2,0.8]),
    ("Stairwell A",  "Stairs fl 0→1",     "place",   1.0,  1.0, 1.75,0, 0.0,   35.0,["stairs","access"],        [0.4,0.4,0.4]),
]

ROBOT_PATH = [
    (0.0,0.0,0.0,0,0.0,0.0),(2.0,0.0,0.0,0,0.0,0.0),(4.0,1.0,0.0,0,90.0,0.0),
    (5.0,2.0,0.0,0,45.0,0.0),(5.0,6.0,0.0,0,90.0,0.0),(0.0,10.0,0.0,0,180.0,0.0),
    (1.0,1.5,0.5,0,0.0,20.0),(1.0,1.0,1.5,0,0.0,30.0),(1.0,0.5,2.5,0,0.0,30.0),
    (1.0,0.2,3.5,1,0.0,0.0),(2.0,2.0,3.5,1,45.0,0.0),(6.0,8.0,3.5,1,45.0,0.0),
]


def sep(title=""):
    w=68
    if title: print(f"\n{'━'*4}  {title}  {'━'*(w-len(title)-7)}")
    else:     print("─"*w)


async def setup(session):
    sep("SETUP")
    entity_ids = {}
    for name,summary,etype,x,y,z,fl,facing,pitch,kwords,eseed in ENTITIES:
        eid = await rm.upsert_entity(
            name=name,summary=summary,entity_type=etype,
            x=x,y=y,z=z,floor_level=fl,
            facing_deg=facing,pitch_deg=pitch,
            tags=[etype],weight=1.0,
            runtime_embedding=make_emb(eseed),
        )
        await rm.add_info_node(
            entity_id=eid,crucial_words=kwords,weight=1.0,
            full_data=f"{name}: {summary}. floor={fl} pos=({x},{y},{z}) facing={facing}° pitch={pitch}°",
            embedding=make_emb(eseed,dim=1536),
        )
        entity_ids[name] = eid
        print(f"  ✓ fl={fl}  {name:<20}  ({x:.0f},{y:.0f},{z:.1f})  hdg={facing:.0f}°")
    for n1,n2,rt,rn in [
        ("Entry Door","Camera 3","monitored_by","Camera covers entry"),
        ("Stairwell A","Office","connects","Stairs to office"),
        ("Stairwell A","Parts Store","connects","Stairs to parts"),
        ("Shelf A1","Shelf A2","adjacent_to","Same aisle"),
    ]:
        await rm.upsert_edge(entity_ids[n1],entity_ids[n2],rel_type=rt,rel_name=rn)
    for x,y,z,fl,hdg,ptch in ROBOT_PATH:
        await rm.record_position(x,y,z,floor_level=fl,heading_deg=hdg,pitch_deg=ptch,session_id=session.session_id)
        for name,_,_,ex,ey,ez,efl,_,_,_,_ in ENTITIES:
            if math.sqrt((x-ex)**2+(y-ey)**2+(z-ez)**2)<1.5:
                log_id=await session.latest_path_log_id()
                await session.log(entity_ids[name],x,y,z,floor_level=fl,path_log_ref=log_id,notes="detected")
    from robot_memory.pathmap import flush_path_to_db
    n,e=await flush_path_to_db(session_id=None)
    print(f"  Flushed: {n} path_nodes  {e} path_edges")
    for name,eid in entity_ids.items():
        try: await rm.anchor_entity_to_map(eid)
        except ValueError: pass
    return entity_ids


async def demo1_cross_floor_path(eids):
    sep("DEMO 1 — Cross-floor 3D path  (Entry Door → Parts Store)")
    result = await rm.think_path(eids["Entry Door"], eids["Parts Store"])
    if result:
        print(f"  Waypoints: {len(result.waypoints)}")
        print(f"  dist_3d  = {result.total_dist_3d_m:.2f} m")
        print(f"  dist_2d  = {result.total_dist_2d_m:.2f} m  (horizontal only)")
        print(f"  ascent   = {result.total_ascent_m:.2f} m ↑")
        print(f"  descent  = {result.total_descent_m:.2f} m ↓")
        trans = result.floor_transitions()
        print(f"  floor transitions: {trans if trans else 'none recorded'}")
        print()
        for i, wp in enumerate(result.waypoints):
            marker = f"fl={wp.floor_level}" if i==0 or result.waypoints[i-1].floor_level!=wp.floor_level else "     "
            print(f"  {i:02d}  ({wp.x:5.1f},{wp.y:5.1f},{wp.z:5.2f})  {marker}  hdg={wp.heading_deg or 0:.0f}°  pitch={wp.pitch_deg or 0:.0f}°")
    else:
        print("  ✗ No path found (check that entities are anchored)")

    print()
    sep("  Charging Bay → Office (floor 0 → floor 1)")
    r2 = await rm.think_path(eids["Charging Bay"], eids["Office"])
    if r2:
        print(f"  ✓  3d={r2.total_dist_3d_m:.2f}m  ascent={r2.total_ascent_m:.2f}m  "
              f"waypoints={len(r2.waypoints)}  transitions={r2.floor_transitions()}")


async def demo2_3d_nearest(eids):
    sep("DEMO 2 — K-nearest (3D) + floor filter + similarity")

    print("\n  [2a] 5 nearest to Shelf A1  all floors  radius=20m")
    nearest = await rm.think_nearest(eids["Shelf A1"], k=5, radius_m=20.0)
    for ent, d3 in nearest:
        emb_ok = "✓emb" if ent.runtime_embedding else "no-emb"
        print(f"  {ent.name:<22}  d3={d3:6.2f}m  floor={ent.floor_level}  z={ent.z:.1f}  {emb_ok}")

    print("\n  [2b] 5 nearest to Shelf A1  floor 0 only")
    nearest_fl0 = await rm.think_nearest(eids["Shelf A1"], k=5, radius_m=20.0, floor_level=0)
    for ent, d3 in nearest_fl0:
        print(f"  {ent.name:<22}  d3={d3:6.2f}m  floor={ent.floor_level}")

    print("\n  [2c] Local think() floor=0  @(5,4,0)  r=5m — similarity filter 'storage'")
    ctx = await rm.think(robot_x=5.0, robot_y=4.0, robot_z=0.0, floor_level=0, radius_m=5.0)
    print(f"  {len(ctx.entities)} entities in 3D window:")
    matched = ctx.filter_by_similarity(Q_STORAGE, min_similarity=0.5)
    print(f"  Storage-similar (sim≥0.5): {[(e.name, f'{s:.3f}') for e,s in matched]}")

    print("\n  [2d] DB-side 3D vector search: 'storage' floor=0 centre=(5,4,0) r=10m")
    sim = await rm.think_similar(Q_STORAGE, top_k=4, radius_m=10.0,
                                  center_x=5.0, center_y=4.0, center_z=0.0,
                                  floor_level=0, min_similarity=0.0)
    for ent, s in sim:
        print(f"  {ent.name:<22}  sim={s:.4f}  floor={ent.floor_level}  xyz=({ent.x:.1f},{ent.y:.1f},{ent.z:.1f})")

    print("\n  [2e] DB-side 3D vector search: 'upper floor / admin' floor=1")
    sim1 = await rm.think_similar(Q_UPPER, top_k=4, floor_level=1, min_similarity=0.0)
    for ent, s in sim1:
        print(f"  {ent.name:<22}  sim={s:.4f}  floor={ent.floor_level}  z={ent.z:.1f}")


async def demo3_local_full_info(eids):
    sep("DEMO 3 — Full info in 3D local window  (think_local_info)")
    for (rx,ry,rz,rfl,label) in [(5.0,4.0,0.0,0,"ground floor centre"),
                                   (3.5,5.0,3.5,1,"mezzanine centre")]:
        ctx = await rm.think_local_info(robot_x=rx,robot_y=ry,robot_z=rz,
                                         floor_level=rfl,radius_m=5.0)
        print(f"\n  Robot @ ({rx},{ry},{rz}) fl={rfl}  [{label}]  "
              f"— {len(ctx.data)} entities in window")
        for eid, d in ctx.data.items():
            e     = d["entity"]
            infos = d["full_info"]
            dist  = math.sqrt((e.x-rx)**2+(e.y-ry)**2+(e.z-rz)**2)
            orient = f"hdg={e.facing_deg:.0f}° pitch={e.pitch_deg or 0:.0f}°"
            bbox   = f"bbox=({e.bbox_dx},{e.bbox_dy},{e.bbox_dz})" if e.bbox_dx else ""
            print(f"  ┌ {e.name}  [{e.entity_type}]  d3={dist:.2f}m  fl={e.floor_level}  {orient}  {bbox}")
            for fi in infos:
                words   = ", ".join(fi.get("crucial_words") or [])
                snippet = (fi.get("full_data") or "")[:90].replace("\n"," / ")
                print(f"  │  words: {words}")
                print(f"  │  data : {snippet}…")
            print(f"  └{'─'*60}")


async def demo4_deep_search(eids):
    sep("DEMO 4 — Deep search (deep_think)")
    dctx = await rm.deep_think()
    print(f"  {dctx.summary()}\n")

    print("  [4a] Per-floor entity breakdown:")
    for fl in sorted(set(e.floor_level for e in dctx.all_entities)):
        ents = dctx.entities_on_floor(fl)
        print(f"    floor {fl}: {[e.name for e in ents]}")

    print("\n  [4b] Keyword search 'storage':")
    for fi in dctx.search_info("storage"):
        print(f"    {fi['entity_id'][:8]}…  {(fi.get('full_data') or '')[:80]}…")

    print("\n  [4c] Keyword search 'stairs':")
    for fi in dctx.search_info("stairs"):
        print(f"    {fi['entity_id'][:8]}…  {(fi.get('full_data') or '')[:80]}…")

    print("\n  [4d] Full graph 3D vector search — 'exit/entry' (no floor filter):")
    sim = await rm.think_similar(Q_EXIT, top_k=4, min_similarity=0.0)
    for ent, s in sim:
        print(f"    {ent.name:<22}  sim={s:.4f}  floor={ent.floor_level}  xyz=({ent.x:.1f},{ent.y:.1f},{ent.z:.1f})")

    print("\n  [4e] 3D bounding-box + orientation in deep context:")
    for e in dctx.all_entities:
        if e.facing_deg is not None:
            bb = f"bbox=({e.bbox_dx},{e.bbox_dy},{e.bbox_dz})" if e.bbox_dx else "no-bbox"
            print(f"    fl={e.floor_level}  {e.name:<22}  hdg={e.facing_deg:.0f}°  {bb}")


async def main():
    sep("ROBOT MEMORY — 3D RETRIEVAL DEMO")
    await rm.init_pool()
    session = await rm.TemporalSession.start()
    eids = await setup(session)
    await demo1_cross_floor_path(eids)
    await demo2_3d_nearest(eids)
    await demo3_local_full_info(eids)
    await demo4_deep_search(eids)
    sep("TEMPORAL SUMMARY")
    await session.dump_summary()
    sep("DONE")
    await rm.close_pool()

if __name__ == "__main__":
    asyncio.run(main())