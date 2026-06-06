**Computation Classification**

`Direct field` means a value read from the current snapshot or a linked entity record. `Aggregates` means reductions over entities in the current snapshot. `Formula` means a fixed calculation or comparison. `Initial-state formula` means a value calculated once from the episode's initial board and reused for every snapshot. `Vectorized geometry` means spatial calculations using only the current snapshot.

`Requires history` means the feature needs one or more earlier snapshots beyond the stored initial board. `Requires configuration` means the value is absent from the minified parquet and must come from environment constants. `Physics/simulation` means fleet-target inference, collision search, or future trajectory simulation.

The classification is based on `data/batch/batch-1.parquet`: snapshots contain 20-44 planets, a median of 11 fleets, and rare late-game outliers up to 1,179 fleets.

**Overall Board Features**

| Feature | Description | Computation |
|---|---|---|
| `current_step` | Current turn number within the game. | Direct field |
| `remaining_steps` | Turns remaining before the game ends. | Formula |
| `game_progress` | Current step divided by maximum game steps. | Formula |
| `player_count` | Number of players active in this game. | Direct field |
| `angular_velocity` | Global orbital rotation speed for moving planets. | Direct field |
| `total_planet_count` | Total number of planets currently on board. | Aggregates |
| `static_planet_count` | Number of planets that never change position. | Aggregates |
| `orbiting_planet_count` | Number of regular planets moving around center. | Aggregates |
| `comet_planet_count` | Number of currently active comet planets. | Aggregates |
| `owned_planet_count` | Number of planets currently owned by us. | Aggregates |
| `enemy_planet_count` | Number of planets owned by all opponents. | Aggregates |
| `neutral_planet_count` | Number of planets currently without an owner. | Aggregates |
| `owned_planets_prod_{1..5}_count` | Owned planet counts for each production level. | Aggregates |
| `enemy_planets_prod_{1..5}_count` | Enemy planet counts for each production level. | Aggregates |
| `neutral_planets_prod_{1..5}_count` | Neutral planet counts for each production level. | Aggregates |
| `owned_production_total` | Combined production rate of all owned planets. | Aggregates |
| `enemy_production_total` | Combined production rate of all enemy planets. | Aggregates |
| `neutral_production_total` | Combined potential production of all neutral planets. | Aggregates |
| `owned_stationed_ships_total` | Total ships currently stationed on owned planets. | Aggregates |
| `enemy_stationed_ships_total` | Total ships currently stationed on enemy planets. | Aggregates |
| `neutral_stationed_ships_total` | Total defending ships stationed on neutral planets. | Aggregates |
| `owned_stationed_ships_mean` | Average stationed ships across all owned planets. | Aggregates |
| `enemy_stationed_ships_mean` | Average stationed ships across all enemy planets. | Aggregates |
| `neutral_stationed_ships_mean` | Average defending ships across all neutral planets. | Aggregates |
| `owned_stationed_ships_max` | Largest ship garrison among our planets. | Aggregates |
| `enemy_stationed_ships_max` | Largest ship garrison among enemy planets. | Aggregates |
| `neutral_stationed_ships_max` | Largest defending garrison among neutral planets. | Aggregates |
| `friendly_active_fleet_count` | Number of our fleets currently travelling. | Aggregates |
| `enemy_active_fleet_count` | Number of opponent fleets currently travelling. | Aggregates |
| `friendly_active_fleet_ships_total` | Total ships travelling inside our active fleets. | Aggregates |
| `enemy_active_fleet_ships_total` | Total ships travelling inside opponent active fleets. | Aggregates |
| `friendly_active_fleet_ships_max` | Ship count of our largest travelling fleet. | Aggregates |
| `enemy_active_fleet_ships_max` | Ship count of largest opponent travelling fleet. | Aggregates |
| `total_ships_controlled` | Owned stationed ships plus friendly travelling ships. | Formula |
| `total_enemy_ships_controlled` | Enemy stationed ships plus enemy travelling ships. | Formula |
| `opponent_{1..3}_is_present` | Mask indicating which player-relative opponent slots exist. | Initial-state formula |
| `opponent_{1..3}_planet_count` | Planet count controlled by each individual opponent slot. | Aggregates |
| `opponent_{1..3}_production_total` | Total production controlled by each individual opponent slot. | Aggregates |
| `opponent_{1..3}_stationed_ships_total` | Stationed ships controlled by each individual opponent slot. | Aggregates |
| `opponent_{1..3}_active_fleet_count` | Travelling fleet count for each individual opponent slot. | Aggregates |
| `opponent_{1..3}_active_fleet_ships_total` | Travelling ships belonging to each individual opponent slot. | Aggregates |

**Single Planet Features**

| Feature | Description | Computation |
|---|---|---|
| `planet_id` | Stable identifier used for source and target outputs. | Direct field |
| `planet_x` | Current horizontal coordinate of the planet center. | Direct field |
| `planet_y` | Current vertical coordinate of the planet center. | Direct field |
| `player_relative_x` | Horizontal coordinate transformed relative to our initial home. | Initial-state formula |
| `player_relative_y` | Vertical coordinate transformed relative to our initial home. | Initial-state formula |
| `planet_radius` | Physical collision radius of this planet. | Direct field |
| `planet_production` | Ships produced by this planet each turn. | Direct field |
| `planet_stationed_ships` | Ships currently stationed directly on this planet. | Direct field |
| `owner_is_ours` | Whether this planet currently belongs to us. | Formula |
| `owner_is_neutral` | Whether this planet currently has no owner. | Formula |
| `owner_is_enemy` | Whether any opponent currently owns this planet. | Formula |
| `owner_opponent_slot` | Player-relative opponent slot currently owning this planet. | Initial-state formula |
| `initial_owner_is_ours` | Whether this planet originally belonged to us. | Initial-state formula |
| `initial_owner_is_enemy` | Whether this planet originally belonged to an opponent. | Initial-state formula |
| `initial_owner_is_neutral` | Whether this planet began without an owner. | Initial-state formula |
| `initial_stationed_ships` | Ships stationed here at the game beginning. | Direct field |
| `is_static` | Whether the planet remains at fixed coordinates. | Formula |
| `is_orbiting` | Whether the planet follows a circular orbit. | Formula |
| `is_comet` | Whether this planet belongs to a comet group. | Formula |
| `orbital_radius` | Planet distance from the board’s central point. | Vectorized geometry |
| `orbital_angle_sin` | Sine representation of current orbital position. | Vectorized geometry |
| `orbital_angle_cos` | Cosine representation of current orbital position. | Vectorized geometry |
| `quadrant_is_home` | Whether planet lies inside our starting quadrant. | Initial-state formula |
| `quadrant_is_opposite` | Whether planet lies diagonally opposite our home. | Initial-state formula |
| `quadrant_is_left_adjacent` | Whether planet occupies our left adjacent quadrant. | Initial-state formula |
| `quadrant_is_right_adjacent` | Whether planet occupies our right adjacent quadrant. | Initial-state formula |
| `is_supplier_frontier` | Whether this planet is the fixed SF logistics anchor. | Initial-state formula |
| `is_attack_frontier` | Whether this planet is the fixed AF attack anchor. | Initial-state formula |
| `is_conductor` | Whether this planet is the fixed supplier for an SF anchor. | Initial-state formula |
| `outgoing_friendly_fleet_count` | Friendly active fleets launched from this planet. | Aggregates |
| `outgoing_friendly_ships_total` | Friendly travelling ships launched from this planet. | Aggregates |
| `outgoing_enemy_fleet_count` | Enemy fleets originally launched from this planet. | Aggregates |
| `outgoing_enemy_ships_total` | Enemy travelling ships launched from this planet. | Aggregates |
| `incoming_friendly_fleet_count` | Friendly fleets predicted to collide with this planet. | Physics/simulation |
| `incoming_friendly_ships_total` | Friendly ships predicted to reach this planet. | Physics/simulation |
| `incoming_enemy_fleet_count` | Enemy fleets predicted to collide with this planet. | Physics/simulation |
| `incoming_enemy_ships_total` | Enemy ships predicted to reach this planet. | Physics/simulation |
| `incoming_enemy_fleet_count_by_player` | Incoming fleet counts separated by individual opponent. | Physics/simulation |
| `incoming_enemy_ships_by_player` | Incoming ship totals separated by individual opponent. | Physics/simulation |
| `incoming_enemy_player_count` | Number of distinct opponents sending fleets toward this planet. | Physics/simulation |
| `incoming_friendly_first_arrival_turn` | Earliest predicted friendly collision with this planet. | Physics/simulation |
| `incoming_enemy_first_arrival_turn` | Earliest predicted enemy collision with this planet. | Physics/simulation |
| `incoming_friendly_last_arrival_turn` | Latest predicted friendly collision within lookahead. | Physics/simulation |
| `incoming_enemy_last_arrival_turn` | Latest predicted enemy collision within lookahead. | Physics/simulation |
| `incoming_net_ship_balance` | Friendly incoming ships minus enemy incoming ships. | Physics/simulation |
| `nearest_owned_planet_distance` | Distance to the closest currently owned planet. | Vectorized geometry |
| `nearest_enemy_planet_distance` | Distance to the closest currently enemy planet. | Vectorized geometry |
| `nearest_neutral_planet_distance` | Distance to the closest currently neutral planet. | Vectorized geometry |
| `distance_from_board_center` | Direct distance from planet to board center. | Vectorized geometry |
| `distance_from_home_center` | Direct distance from planet to home-quadrant center. | Initial-state formula |
| `distance_from_opposite_center` | Direct distance from planet to opposite-quadrant center. | Initial-state formula |

**Individual Active Fleet Features**

| Feature | Description | Computation |
|---|---|---|
| `fleet_id` | Stable fleet identifier retained only as metadata. | Direct field |
| `fleet_x` | Current horizontal coordinate of the travelling fleet. | Direct field |
| `fleet_y` | Current vertical coordinate of the travelling fleet. | Direct field |
| `fleet_player_relative_x` | Horizontal fleet coordinate relative to our initial home. | Initial-state formula |
| `fleet_player_relative_y` | Vertical fleet coordinate relative to our initial home. | Initial-state formula |
| `fleet_ship_count` | Number of ships contained inside this fleet. | Direct field |
| `fleet_speed` | Movement speed calculated directly from fleet ship count. | Formula |
| `fleet_angle_sin` | Sine representation of the fleet movement direction. | Formula |
| `fleet_angle_cos` | Cosine representation of the fleet movement direction. | Formula |
| `fleet_velocity_x` | Horizontal fleet velocity calculated from speed and angle. | Vectorized geometry |
| `fleet_velocity_y` | Vertical fleet velocity calculated from speed and angle. | Vectorized geometry |
| `fleet_owner_is_ours` | Whether this fleet belongs to our player. | Formula |
| `fleet_owner_is_enemy` | Whether this fleet belongs to any opponent. | Formula |
| `fleet_owner_opponent_slot` | Player-relative opponent slot controlling this fleet. | Initial-state formula |
| `fleet_source_planet_id` | Launching planet identifier retained only as metadata. | Direct field |
| `fleet_source_is_ours` | Whether the source planet currently belongs to us. | Formula |
| `fleet_source_is_neutral` | Whether the source planet is currently neutral. | Formula |
| `fleet_source_is_enemy` | Whether an opponent currently owns the source planet. | Formula |
| `fleet_source_production` | Production rate of the fleet's launching planet. | Direct field |
| `fleet_source_current_ships` | Ships currently stationed on the launching planet. | Direct field |
| `fleet_distance_to_source` | Current distance between fleet and launching planet. | Vectorized geometry |
| `fleet_distance_from_board_center` | Current fleet distance from the board center. | Vectorized geometry |
| `fleet_quadrant_is_home` | Whether fleet currently occupies our home quadrant. | Initial-state formula |
| `fleet_quadrant_is_opposite` | Whether fleet currently occupies the opposite quadrant. | Initial-state formula |
| `fleet_quadrant_is_left_adjacent` | Whether fleet occupies our left adjacent quadrant. | Initial-state formula |
| `fleet_quadrant_is_right_adjacent` | Whether fleet occupies our right adjacent quadrant. | Initial-state formula |
| `fleet_distance_to_owned_centroid` | Fleet distance from our current territory center. | Vectorized geometry |
| `fleet_distance_to_enemy_centroid` | Fleet distance from enemy territory center. | Vectorized geometry |
| `fleet_nearest_planet_distance` | Distance from fleet to its nearest planet. | Vectorized geometry |
| `fleet_nearest_owned_planet_distance` | Distance from fleet to nearest owned planet. | Vectorized geometry |
| `fleet_nearest_enemy_planet_distance` | Distance from fleet to nearest enemy planet. | Vectorized geometry |
| `fleet_nearest_neutral_planet_distance` | Distance from fleet to nearest neutral planet. | Vectorized geometry |
| `fleet_has_predicted_hit` | Whether the fleet is predicted to collide with a planet. | Physics/simulation |
| `fleet_predicted_target_id` | Predicted target identifier retained only for entity linking. | Physics/simulation |
| `fleet_predicted_collision_turns` | Turns remaining until the predicted collision. | Physics/simulation |
| `fleet_predicted_collision_distance` | Remaining fleet path distance before predicted collision. | Physics/simulation |
| `fleet_target_owner_is_friendly` | Whether the predicted target is friendly to the fleet owner. | Physics/simulation |
| `fleet_target_owner_is_neutral` | Whether the predicted target is currently neutral. | Physics/simulation |
| `fleet_target_owner_is_hostile` | Whether the predicted target is hostile to the fleet owner. | Physics/simulation |
| `fleet_target_ship_count` | Current stationed ships on the predicted target. | Physics/simulation |
| `fleet_target_production` | Production rate of the predicted target. | Physics/simulation |
| `fleet_target_is_source` | Whether the predicted target originally launched this fleet. | Physics/simulation |

**Planet-To-Planet Relationship Features**

These channels are computed for each ordered planet pair and supplied to attention as relationship information. The number of channels remains fixed even when the number of planets changes.

| Feature | Description | Computation |
|---|---|---|
| `planet_pair_delta_x` | Horizontal displacement from source planet to target planet. | Vectorized geometry |
| `planet_pair_delta_y` | Vertical displacement from source planet to target planet. | Vectorized geometry |
| `planet_pair_distance` | Center-to-center distance between both planets. | Vectorized geometry |
| `planet_pair_surface_distance` | Distance remaining after subtracting both planet radii. | Vectorized geometry |
| `planet_pair_direction_sin` | Sine representation of source-to-target direction. | Vectorized geometry |
| `planet_pair_direction_cos` | Cosine representation of source-to-target direction. | Vectorized geometry |
| `planet_pair_relative_velocity_x` | Target horizontal velocity minus source horizontal velocity. | Vectorized geometry |
| `planet_pair_relative_velocity_y` | Target vertical velocity minus source vertical velocity. | Vectorized geometry |
| `planet_pair_relative_speed` | Magnitude of relative movement between both planets. | Vectorized geometry |
| `planet_pair_same_owner` | Whether both planets currently share the same owner. | Formula |
| `planet_pair_both_owned` | Whether both planets currently belong to us. | Formula |
| `planet_pair_source_owned_target_neutral` | Whether pair represents our source and neutral target. | Formula |
| `planet_pair_source_owned_target_enemy` | Whether pair represents our source and enemy target. | Formula |
| `planet_pair_same_quadrant` | Whether both planets occupy the same relative quadrant. | Formula |
| `planet_pair_production_difference` | Target production minus source production. | Formula |
| `planet_pair_ship_difference` | Source stationed ships minus target stationed ships. | Formula |
| `planet_pair_ship_ratio` | Source stationed ships divided by target stationed ships. | Formula |
| `planet_pair_target_incoming_friendly_ships` | Friendly ships already travelling toward target. | Physics/simulation |
| `planet_pair_target_incoming_enemy_ships` | Enemy ships already travelling toward target. | Physics/simulation |

**Fleet-To-Planet Relationship Features**

These channels describe every active fleet relative to every planet. Collision-derived channels reuse the exact predicted fleet target and arrival turn.

| Feature | Description | Computation |
|---|---|---|
| `fleet_planet_delta_x` | Horizontal displacement from fleet to planet. | Vectorized geometry |
| `fleet_planet_delta_y` | Vertical displacement from fleet to planet. | Vectorized geometry |
| `fleet_planet_distance` | Center-to-center distance from fleet to planet. | Vectorized geometry |
| `fleet_planet_surface_distance` | Distance remaining after subtracting planet radius. | Vectorized geometry |
| `fleet_planet_direction_sin` | Sine representation of fleet-to-planet direction. | Vectorized geometry |
| `fleet_planet_direction_cos` | Cosine representation of fleet-to-planet direction. | Vectorized geometry |
| `fleet_planet_heading_alignment` | Alignment between fleet heading and planet direction. | Vectorized geometry |
| `fleet_planet_cross_track_offset` | Perpendicular offset from fleet's straight flight line. | Vectorized geometry |
| `fleet_planet_relative_velocity_x` | Planet horizontal velocity minus fleet horizontal velocity. | Vectorized geometry |
| `fleet_planet_relative_velocity_y` | Planet vertical velocity minus fleet vertical velocity. | Vectorized geometry |
| `fleet_planet_relative_speed` | Relative movement speed between fleet and planet. | Vectorized geometry |
| `fleet_planet_static_eta` | Straight-line distance divided by current fleet speed. | Vectorized geometry |
| `fleet_planet_owner_matches` | Whether fleet and planet currently share an owner. | Formula |
| `fleet_planet_is_source` | Whether planet originally launched this fleet. | Formula |
| `fleet_planet_is_friendly_destination` | Whether planet is friendly to the fleet owner. | Formula |
| `fleet_planet_is_hostile_destination` | Whether planet is hostile to the fleet owner. | Formula |
| `fleet_planet_is_predicted_destination` | Whether this planet is the fleet's predicted collision target. | Physics/simulation |
| `fleet_planet_collision_eta` | Collision ETA when this planet is the predicted target. | Physics/simulation |
| `fleet_planet_arrives_before_friendly` | Whether fleet arrives before other friendly fleets at target. | Physics/simulation |
| `fleet_planet_arrives_before_enemy` | Whether fleet arrives before opposing fleets at target. | Physics/simulation |
| `fleet_planet_same_turn_friendly_ships` | Friendly ships predicted to arrive on the same turn. | Physics/simulation |
| `fleet_planet_same_turn_enemy_ships` | Enemy ships predicted to arrive on the same turn. | Physics/simulation |

**Comet Features**

| Feature | Description | Computation |
|---|---|---|
| `comet_group_id` | Stable comet-group identifier retained only as metadata. | Direct field |
| `comet_path_index` | Current discrete position within the comet path. | Direct field |
| `comet_path_length` | Total number of positions in the active comet path. | Aggregates |
| `comet_path_progress` | Current path index divided by full path length. | Formula |
| `comet_velocity_x` | Current horizontal comet movement per turn. | Vectorized geometry |
| `comet_velocity_y` | Current vertical comet movement per turn. | Vectorized geometry |
| `comet_speed_current` | Magnitude of current comet movement velocity. | Vectorized geometry |

**Model Input Rules**

`planet_id`, `fleet_id`, `fleet_source_planet_id`, `fleet_predicted_target_id`, and `comet_group_id` are metadata used for linking and output reconstruction. They must not be treated as continuous numeric model features.

All coordinates, distances, ship counts, speeds, production values, and time values should be normalized before entering the model.

Missing group statistics should use a mask plus a neutral numeric value, rather than an arbitrary sentinel that could be mistaken for real data.

The same feature widths and entity schemas are used for both two-player and four-player games. `player_count` and `opponent_{1..3}_is_present` identify the active mode and valid opponent slots.

Rotate each episode into a canonical player-relative orientation using the initial home quadrant. Assign opponent slots by initial relative quadrant:

- `opponent_1`: left-adjacent quadrant
- `opponent_2`: diagonally opposite quadrant
- `opponent_3`: right-adjacent quadrant

In two-player games, only the opposite opponent slot is present; unused slots are zero-filled and masked. In four-player games, all three opponent slots are present. Arbitrary environment player IDs must never determine slot meaning.

Global `enemy_*` features aggregate all opponents. Per-opponent features retain the three fixed slots so the model can distinguish pressure and ownership belonging to different players.

SF, AF, and C are fixed geometric logistics roles calculated once from the initial board. They do not identify a particular opponent; this keeps their meaning valid in both two-player and four-player games.

Angles must be represented using sine and cosine rather than raw wrapped radians.

Variable planet and fleet counts are represented by padded token tensors and masks. Each token type still has a fixed feature width.
