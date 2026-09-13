# Road-network instructions

## Scope

This directory owns the shared OSM/SUMO network and traffic-light definitions
used by matching, simulation, and flow analysis. A change here can invalidate
every downstream result. Follow the repository-root instructions as well.

## File roles

- `download_osm.py` fetches the selected geographic area and produces the OSM
  XML/GeoJSON source data.
- `basemap.osm.xml` and `basemap.geojson` retain OSM identities and geometry.
- `net_tls.net.xml` is the canonical SUMO network used by all downstream code.
  It is normally generated with `netconvert`, not maintained by hand.
- `find_reverse_edge_pairs.py` derives `reverse_edge_pairs.txt` for simulation
  route cleanup.
- `tls_schedule.add.xml` and the peak-period XML files define experiment-specific
  signal programs/schedules and must reference IDs present in the canonical net.

## Network invariants

- Preserve OSM/SUMO/junction/lane/TLS IDs exactly as strings. Signed edge IDs
  encode direction; `#` suffixes identify split segments; `:` prefixes identify
  internal edges.
- Do not globally reformat or manually patch the large generated net XML for a
  source-level change. Update the OSM input, `netconvert` command/options, or
  post-processing generator and regenerate intentionally.
- Passenger reachability and `<connection from=... to=...>` relationships are
  part of the matching and simulation contract.
- When changing the network, verify that route edges, TLS IDs, controlled-link
  indexes, additional signal programs, and propagation-link edge IDs still
  exist.
- Regenerate the reverse-edge list whenever edge IDs or topology change.
- TLS phase-state strings must have the same width and link-index ordering as
  the controlled connections. Preserve both protected and permissive green
  semantics (`G` and `g`).
- Keep the bounding box and Overpass query explicit. Downloading new OSM data
  changes the experiment's source data and requires user intent and network
  access.

## Validation

- For XML-only edits, parse every changed file and verify referenced IDs against
  `net_tls.net.xml`.
- For topology changes, run `netconvert`/SUMO validation when available, then
  rebuild the edge graph and reverse pairs and test representative routes.
- Inspect `netconvert` and SUMO logs for discarded connections, invalid TLS link
  indexes, unreachable routes, and permission changes.
- Do not run `download_osm.py` or overwrite `net_tls.net.xml` as a routine test.
  These operations are network-dependent and can replace canonical data.
