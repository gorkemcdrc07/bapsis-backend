import pandas as pd
import numpy as np
from ortools.constraint_solver import pywrapcp, routing_enums_pb2


class TariffEngine:
    def __init__(self, data_file, demand_file):
        self.data_file = data_file
        self.demand_file = demand_file
        self._load_data()
        self.params = {"max_stops_per_truck": 3, "vehicle_fixed_cost": 0}
        self.last_truck_loads = {}
        self.missing_tariffs = set()

    def _load_data(self):
        self.wh = pd.read_excel(self.data_file, sheet_name="Warehouses")
        self.trucks = pd.read_excel(self.data_file, sheet_name="Trucks")
        self.cross = pd.read_excel(self.data_file, sheet_name="CrossDistance", index_col=0)
        self.demands = pd.read_excel(self.demand_file, sheet_name="Demands")

        try:
            self.groups = pd.read_excel(self.data_file, sheet_name="ConsolidationGroups")
        except Exception:
            self.groups = pd.DataFrame(columns=["group", "wh_name"])

        try:
            self.tariffs = pd.read_excel(self.data_file, sheet_name="Tariffs")
        except Exception:
            self.tariffs = pd.DataFrame(
                columns=[
                    "last_stop",
                    "base_price",
                    "extra_km_price",
                    "extra_stop_same_city",
                    "extra_stop_diff_city",
                ]
            )

        if "capacity" in self.trucks.columns:
            self.truck_capacity = int(self.trucks["capacity"].max())
        else:
            self.truck_capacity = 33

        self.wh_by_name = self.wh.set_index("wh_name").T.to_dict()

        depot_row = self.wh[self.wh.get("is_origin", 0) == 1]
        if depot_row.empty:
            raise ValueError("No depot defined (set is_origin=1 in Warehouses sheet)")
        self.depot_name = depot_row["wh_name"].iloc[0]

    def _split_large_demands(self, df):
        rows = []

        if df is None or df.empty:
            return pd.DataFrame(columns=["wh_name", "demand_units"])

        for _, row in df.iterrows():
            wh_name = row["wh_name"]
            demand = int(row["demand_units"])

            while demand > self.truck_capacity:
                rows.append({
                    "wh_name": wh_name,
                    "demand_units": self.truck_capacity
                })
                demand -= self.truck_capacity

            if demand > 0:
                rows.append({
                    "wh_name": wh_name,
                    "demand_units": demand
                })

        return pd.DataFrame(rows, columns=["wh_name", "demand_units"])

    def _demands_for_date(self, date_str):
        df = self.demands.copy()

        df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.strftime("%Y-%m-%d")
        date_str = pd.to_datetime(date_str, errors="coerce").strftime("%Y-%m-%d")

        df = df[df["date"] == date_str].copy()
        df = self._split_large_demands(df)

        if df.empty:
            return pd.DataFrame(columns=["wh_name", "demand_units", "group"])

        if self.groups is not None and not self.groups.empty and "wh_name" in self.groups.columns:
            grp = self.groups.copy()

            if "group" not in grp.columns:
                grp["group"] = grp["wh_name"]

            grp = grp[["wh_name", "group"]].drop_duplicates()
            df = df.merge(grp, on="wh_name", how="left")
            df["group"] = df["group"].fillna(df["wh_name"])
        else:
            df["group"] = df["wh_name"]

        return df

    def allocate_greedy(self, date_str):
        df = self._demands_for_date(date_str)
        plan = {}
        truck_no = 1
        self.last_truck_loads = {}

        for gid, gdf in df.groupby("group"):
            pallets = gdf["demand_units"].sum()
            if pallets <= self.truck_capacity:
                plan[f"Truck{truck_no}"] = list(gdf["wh_name"])
                self.last_truck_loads[f"Truck{truck_no}"] = int(pallets)
                truck_no += 1
            else:
                remaining = gdf.copy()

                while not remaining.empty:
                    load = 0
                    stops = []
                    picked_idx = []

                    for idx, row in remaining.iterrows():
                        wh_name, dem = row["wh_name"], int(row["demand_units"])
                        if load + dem <= self.truck_capacity:
                            stops.append(wh_name)
                            load += dem
                            picked_idx.append(idx)

                        if load == self.truck_capacity:
                            break

                    if not picked_idx:
                        raise ValueError(
                            f"Greedy could not allocate remaining rows for group '{gid}'. "
                            f"Remaining sample: {remaining.head(5).to_dict('records')}"
                        )

                    plan[f"Truck{truck_no}"] = stops
                    self.last_truck_loads[f"Truck{truck_no}"] = int(load)
                    truck_no += 1

                    remaining = remaining.drop(index=picked_idx)

        return plan

    def optimize_with_ortools(self, date_str):
        df = self._demands_for_date(date_str)
        if df.empty:
            return {}

        plan = {}
        self.last_truck_loads = {}
        truck_no = 1
        max_stops = int(self.params.get("max_stops_per_truck", 3))
        fixed_cost = int(self.params.get("vehicle_fixed_cost", 0))

        for gid, gdf in df.groupby("group"):
            locs = [self.depot_name] + list(gdf["wh_name"])
            demands = [0] + [int(x) for x in gdf["demand_units"]]
            n = len(locs)

            dist_matrix = np.zeros((n, n), dtype=int)
            for i, a in enumerate(locs):
                for j, b in enumerate(locs):
                    if i != j:
                        try:
                            dist_matrix[i, j] = int(self.cross.loc[a, b])
                        except Exception:
                            dist_matrix[i, j] = 999999

            num_vehicles = sum((d + self.truck_capacity - 1) // self.truck_capacity for d in demands)
            num_vehicles = max(1, num_vehicles)
            depot_index = 0

            manager = pywrapcp.RoutingIndexManager(n, num_vehicles, depot_index)
            routing = pywrapcp.RoutingModel(manager)

            def distance_cb(i, j):
                return int(dist_matrix[manager.IndexToNode(i)][manager.IndexToNode(j)])

            transit_idx = routing.RegisterTransitCallback(distance_cb)
            routing.SetArcCostEvaluatorOfAllVehicles(transit_idx)

            if fixed_cost > 0:
                routing.SetFixedCostOfAllVehicles(fixed_cost)

            def demand_cb(i):
                return int(demands[manager.IndexToNode(i)])

            demand_idx = routing.RegisterUnaryTransitCallback(demand_cb)
            routing.AddDimensionWithVehicleCapacity(
                demand_idx, 0, [self.truck_capacity] * num_vehicles, True, "Capacity"
            )

            if max_stops > 0:
                def stop_cb(i):
                    return 0 if manager.IndexToNode(i) == depot_index else 1

                stop_idx = routing.RegisterUnaryTransitCallback(stop_cb)
                routing.AddDimensionWithVehicleCapacity(
                    stop_idx, 0, [max_stops] * num_vehicles, True, "Stops"
                )

            search = pywrapcp.DefaultRoutingSearchParameters()
            search.first_solution_strategy = routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
            search.local_search_metaheuristic = routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH
            search.time_limit.seconds = 10

            solution = routing.SolveWithParameters(search)
            if not solution:
                continue

            for v in range(num_vehicles):
                idx = routing.Start(v)
                route, load = [], 0
                while not routing.IsEnd(idx):
                    node = manager.IndexToNode(idx)
                    if node != depot_index:
                        route.append(locs[node])
                        load += demands[node]
                    idx = solution.Value(routing.NextVar(idx))
                if route:
                    plan[f"Truck{truck_no}"] = route
                    self.last_truck_loads[f"Truck{truck_no}"] = int(load)
                    truck_no += 1

        return plan

    def _calc_cost_components(self, stops):
        if not stops:
            return {
                "base_cost": 0.0,
                "add_stop_cost": 0.0,
                "extra_km_price": 0.0,
                "baseline_km": 0.0,
                "route_km": 0.0,
                "extra_km_over_baseline": 0.0,
                "extra_km_chargeable": 0.0,
                "extra_km_cost": 0.0,
                "total_cost": 0.0,
            }

        last_wh = stops[-1]
        row = self.tariffs[self.tariffs["last_stop"] == last_wh]

        if row.empty:
            self.missing_tariffs.add(str(last_wh))
            return {
                "base_cost": 1000.0,
                "add_stop_cost": 0.0,
                "extra_km_price": 0.0,
                "baseline_km": 0.0,
                "route_km": 0.0,
                "extra_km_over_baseline": 0.0,
                "extra_km_chargeable": 0.0,
                "extra_km_cost": 0.0,
                "total_cost": 1000.0,
            }

        base_cost = float(row["base_price"].iloc[0])
        extra_same = float(row.get("extra_stop_same_city", 0).iloc[0])
        extra_diff = float(row.get("extra_stop_diff_city", 0).iloc[0])
        extra_km_price = float(row.get("extra_km_price", 0).iloc[0]) if "extra_km_price" in row.columns else 0.0

        add_stop_cost = 0.0
        for stop in stops[:-1]:
            same_city = self.wh_by_name[stop]["city"] == self.wh_by_name[last_wh]["city"]
            add_stop_cost += extra_same if same_city else extra_diff

        baseline_km = 0.0
        route_km = 0.0
        extra_km_over_baseline = 0.0
        extra_km_chargeable = 0.0
        extra_km_cost = 0.0

        if len(stops) >= 2 and extra_km_price > 0:
            def dist(a, b):
                try:
                    return float(self.cross.loc[a, b])
                except Exception:
                    return None

            baseline = dist(self.depot_name, last_wh)

            ok = True
            prev = self.depot_name
            total_route = 0.0

            for s in stops:
                d = dist(prev, s)
                if d is None:
                    ok = False
                    break
                total_route += d
                prev = s

            if ok and baseline is not None:
                baseline_km = float(baseline)
                route_km = float(total_route)
                extra_km_over_baseline = route_km - baseline_km
                extra_km_chargeable = extra_km_over_baseline - 50.0
                if extra_km_chargeable > 0:
                    extra_km_cost = extra_km_chargeable * extra_km_price
                else:
                    extra_km_chargeable = 0.0
                    extra_km_cost = 0.0

        total_cost = base_cost + add_stop_cost + extra_km_cost

        return {
            "base_cost": float(base_cost),
            "add_stop_cost": float(add_stop_cost),
            "extra_km_price": float(extra_km_price),
            "baseline_km": float(baseline_km),
            "route_km": float(route_km),
            "extra_km_over_baseline": float(extra_km_over_baseline),
            "extra_km_chargeable": float(extra_km_chargeable),
            "extra_km_cost": float(extra_km_cost),
            "total_cost": float(total_cost),
        }

    def _calc_cost(self, stops):
        return self._calc_cost_components(stops)["total_cost"]

    def multi_truck_costs(self, plan):
        rows = []
        for truck, stops in plan.items():
            pallets = self.last_truck_loads.get(truck, 0)
            comp = self._calc_cost_components(stops)

            rows.append({
                "truck": truck,
                "stops": len(stops),
                "total_pallets": pallets,
                "utilization_%": round(100 * pallets / self.truck_capacity, 1),
                "base_cost": round(comp["base_cost"], 2),
                "add_stop_cost": round(comp["add_stop_cost"], 2),
                "extra_km_price": round(comp["extra_km_price"], 4),
                "baseline_km": round(comp["baseline_km"], 2),
                "route_km": round(comp["route_km"], 2),
                "extra_km_over_baseline": round(comp["extra_km_over_baseline"], 2),
                "extra_km_chargeable": round(comp["extra_km_chargeable"], 2),
                "extra_km_cost": round(comp["extra_km_cost"], 2),
                "total_cost": round(comp["total_cost"], 2),
                "route": " → ".join(stops),
            })

        return pd.DataFrame(rows)

    def _order_stops_by_depot(self, stops):
        return stops