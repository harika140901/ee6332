import datetime
import re, sys
import networkx as nx
import numpy as np
from gpkit import Variable, VectorVariable, Model
sys.setrecursionlimit(1000000)
class Gate:
    def __init__(self, name, gate_type, p=0, l=0):
        self.name, self.gate_type, self.inputs, self.outputs = name, gate_type, [], []
        self.p = p
        self.l = l

    def add_input(self, gate):
        if gate not in self.inputs:
            self.inputs.append(gate)
            if self not in gate.outputs: gate.outputs.append(self)

    def add_output(self, gate):
        if gate not in self.outputs:
            self.outputs.append(gate)
            if self not in gate.inputs: gate.inputs.append(self)

class Net:
    def __init__(self, name, source=None, destinations=None):
        self.name, self.source, self.destinations, self.value = name, source, destinations or [], None

    def add_destination(self, gate):
        if gate not in self.destinations: self.destinations.append(gate)

    def set_source(self, gate):
        self.source = gate

class Circuit:
    def __init__(self):
        self.gates, self.nets, self.inputs, self.outputs = {}, {}, [], []
        self.gate_order = []

    def add_gate(self, gate):
        self.gates[gate.name] = gate
        if gate.name not in self.gate_order:
            self.gate_order.append(gate.name)

    def add_net(self, net): self.nets[net.name] = net

    def connect(self, source_gate_name, dest_gate_name, net_name=None):
        source_gate, dest_gate = self.gates.get(source_gate_name), self.gates.get(dest_gate_name)
        if not source_gate or not dest_gate: raise ValueError(f"Cannot connect: gate not found")

        net_name = net_name or f"{source_gate_name}_to_{dest_gate_name}"
        if net_name not in self.nets:
            net = Net(net_name, source_gate, [dest_gate])
            self.add_net(net)
        else:
            net = self.nets[net_name]
            net.set_source(source_gate)
            net.add_destination(dest_gate)

        source_gate.add_output(dest_gate)
        dest_gate.add_input(source_gate)
        return net

    def set_input_gates(self, gate_names): self.inputs = gate_names

    def set_output_gates(self, gate_names): self.outputs = gate_names

    def get_fanouts(self):
        return {gate_name: [g.name for g in gate.outputs] for gate_name, gate in self.gates.items()}
    
    def get_all_paths(self):
        sources = [gn for gn, g in self.gates.items() if not g.inputs]
        sinks = [gn for gn, g in self.gates.items() if not g.outputs]
        all_paths = []

        def dfs(current, path):
            path.append(current)
            if current in sinks:
                all_paths.append(path.copy())
            else:
                for next_gate in self.gates[current].outputs:
                    dfs(next_gate.name, path)
            path.pop()

        for source in sources:
            dfs(source, [])

        return all_paths

    def get_longest_paths(self):
        sources = [gn for gn, g in self.gates.items() if not g.inputs]
        sinks = [gn for gn, g in self.gates.items() if not g.outputs]
        longest_paths, max_length = [], 0

        def find_paths(current, path, visited):
            nonlocal longest_paths, max_length
            visited.add(current); path.append(current)

            if current in sinks:
                path_length = len(path)
                if path_length > max_length: max_length, longest_paths = path_length, [path.copy()]
                elif path_length == max_length: longest_paths.append(path.copy())

            for next_gate in self.gates[current].outputs:
                if next_gate.name not in visited: find_paths(next_gate.name, path, visited)

            path.pop(); visited.remove(current)

        for source in sources: find_paths(source, [], set())
        return longest_paths, max_length

def parse(file_contents):
    circuit = Circuit()
    patterns = {
        'input': re.compile(r'input\s+([^;]+);'),
        'output': re.compile(r'output\s+([^;]+);'),
        'wire': re.compile(r'wire\s+([^;]+);'),
        'gate': re.compile(r'(not|nand|nor)\s+(\w+)\s*\(([^)]+)\);', re.MULTILINE)
    }

    inputs = [inp.strip() for inp in patterns['input'].search(file_contents).group(1).split(',')] if patterns['input'].search(file_contents) else []
    outputs = [out.strip() for out in patterns['output'].search(file_contents).group(1).split(',')] if patterns['output'].search(file_contents) else []
    wires = [wire.strip() for wire in patterns['wire'].search(file_contents).group(1).split(',')] if patterns['wire'].search(file_contents) else []

    for signal_name in inputs + outputs + wires: circuit.add_net(Net(signal_name))

    for gate_match in patterns['gate'].finditer(file_contents):
        gate_type, gate_name = gate_match.group(1).upper(), gate_match.group(2)
        if not any(gate_name.lower().startswith(prefix) for prefix in ['nand', 'nor', 'not']): continue

        ports = [port.strip() for port in gate_match.group(3).split(',')]
        output_port, input_ports = ports[0], ports[1:]

        gate = Gate(gate_name, gate_type)
        circuit.add_gate(gate)

        output_net = circuit.nets.get(output_port) or Net(output_port)
        if output_port not in circuit.nets: circuit.add_net(output_net)
        output_net.set_source(gate)

        for input_port in input_ports:
            input_net = circuit.nets.get(input_port) or Net(input_port)
            if input_port not in circuit.nets: circuit.add_net(input_net)
            input_net.add_destination(gate)

        if output_port in outputs and circuit.gates.get(output_port): circuit.connect(gate_name, output_port)

    for net_name, net in circuit.nets.items():
        if net.source and net.destinations:
            source_gate = net.source
            for dest_gate in net.destinations:
                if source_gate.name != dest_gate.name:
                    is_gate = lambda g: any(g.name.lower().startswith(p) for p in ['nand', 'nor', 'not'])
                    if is_gate(source_gate) and is_gate(dest_gate): circuit.connect(source_gate.name, dest_gate.name, net_name)

    circuit.set_input_gates(inputs)
    circuit.set_output_gates(outputs)
    return circuit

def gates_with_primary_inputs(circuit):
    result = []
    for net_name in circuit.inputs:
        net = circuit.nets.get(net_name)
        if net:
            for gate in net.destinations:
                result.append(gate.name)
    return sorted(set(result))

def gates_with_primary_outputs(circuit):
    result = []
    for net_name in circuit.outputs:
        net = circuit.nets.get(net_name)
        if net and net.source:
            result.append(net.source.name)
    for gate_name, gate in circuit.gates.items():
        if not gate.outputs:
            result.append(gate_name)
    return sorted(set(result))

def analyze_circuit(ckt):
    def get_gate_number(gate_name):
        parts = gate_name.split('_')
        if len(parts) < 2:
            return float('inf')
        num_part = parts[1]
        return int(num_part)

    fanouts_dict = {}
    fanouts = ckt.get_fanouts()
    for gate_name in sorted(fanouts.keys(), key=get_gate_number):
        gate_fanouts = fanouts[gate_name]
        sorted_fanouts = sorted(gate_fanouts, key=get_gate_number) if gate_fanouts else []
        fanouts_dict[gate_name] = sorted_fanouts

    critical_paths_dict = {
        "max_length": 0,
        "paths": []
    }
    longest_paths, max_length = ckt.get_longest_paths()
    if longest_paths:
        critical_paths_dict["max_length"] = max_length
        def get_path_priority(path):
            if not path: return float('inf')
            return get_gate_number(path[0])
        sorted_paths = sorted(longest_paths, key=get_path_priority)
        critical_paths_dict["paths"] = sorted_paths
    return fanouts_dict, critical_paths_dict

logicalEffortDict = {
    ("NAND", 2): 4/3,
    ("NAND", 3): 5/3,
    ("NAND", 4): 2,
    ("NOR", 2): 5/3,
    ("NOR", 3): 7/3,
    ("NOR", 4): 3,
    ("INV", 1): 1,
    ("NOT", 1): 1
}

twall = {
    "c17.txt": 28.83,
    "c432.txt": 153.47,
    "c880.txt": 118.93,
    "c1908.txt": 177.58,
    "c2670.txt": 178.69,
    "c3540.txt": 220.71,
    "c5315.txt": 203.07,
    "c6288.txt": 564.64,
    "c7552.txt": 167.86

}
cload = 1000
max_gate_size = 64

def get_gate_type_and_inputs(gate_name):
    gatenum = gate_name.split("_")[1]
    prefix = gate_name.split("_")[0]
    gate_type = prefix[:-1]
    num_inputs = int(prefix[-1])
    return gate_type.upper(), num_inputs, gatenum

def gates_with_all_primary_inputs(circuit):
    primary_nets = set(circuit.inputs)
    qualifying_gates = []
    
    for gate_name, gate in circuit.gates.items():
        # Find all nets feeding into this gate
        input_nets = [
            net_name for net_name, net in circuit.nets.items()
            if gate in net.destinations
        ]
        
        # Check if all input nets are primary inputs
        if all(net in primary_nets for net in input_nets) and input_nets:
            qualifying_gates.append(gate_name)
    
    return sorted(qualifying_gates)

import numpy as np
from gpkit import Variable, VectorVariable, Model

def minimum_area_node_based(fanouts_dict, Tspec, inpGates, x, newlimits):
    
    N = len(fanouts_dict)
    Gsize = VectorVariable(N, "x", sign="positive")

    T = VectorVariable(N, "T", sign="positive")
    
    g = [1.0] * N
    p = [2.0] * N  
    gate_index_map = {}

    for gate in fanouts_dict:
        gate_type, num_inputs, _ = get_gate_type_and_inputs(gate)
        idx = int(gate.split("_")[1]) - 1
        gate_index_map[gate] = idx
        g[idx] = logicalEffortDict.get((gate_type, num_inputs), 1.0)
        p[idx] = num_inputs
    
    constraints = []
    
    for (name,symbol,val) in newlimits:
        newindex = gate_index_map[name]
        if symbol == '<=':
            constraints.append(Gsize[newindex] <= val)
        else:
            constraints.append(Gsize[newindex] >= val)

    for net in circuit.nets:
        total_cap = 0
        for gate in circuit.nets[net].destinations:
            if gate.name in inpGates and circuit.nets[net].source is None:
                idx = gate_index_map[gate.name]
                total_cap += g[idx] * Gsize[idx]
        if total_cap != 0:
            constraints.append(total_cap <= 50)

    primary_input_gates = []
    for gate_name in inpGates:
        if gate_name in gate_index_map and gate_name in x:
            idx = gate_index_map[gate_name]
            T[idx] = 0
            primary_input_gates.append(gate_name)
    

    for gate in fanouts_dict:
        idx = gate_index_map[gate]
        input_gates = circuit.gates[gate].outputs
        

        total_fo_effort = 0
        for fanout in fanouts_dict[gate]:
            if fanout in gate_index_map:
                f_idx = gate_index_map[fanout]
                total_fo_effort += g[f_idx] * Gsize[f_idx]
        
        delay = p[idx]
        if total_fo_effort != 0:
            delay += total_fo_effort / Gsize[idx]
        
        for input_gate in input_gates:
            input_idx = gate_index_map[input_gate.name]
            constraints.append(T[input_idx] >= T[idx] + delay)

    output_gates = gates_with_primary_outputs(circuit)
    for gate in output_gates:
        idx = gate_index_map[gate]
        load_delay = cload / Gsize[idx]
        constraints.append(T[idx] + load_delay + p[idx]<= Tspec)

    constraints += [Gsize >= 1, Gsize <= max_gate_size]
    objective = sum(Gsize)
    m = Model(objective, constraints)

    try:
        sol = m.solve(verbosity=0)
        GsizeList = [round(float(x), 3) for x in sol["variables"]["x"]]
        keys = list(fanouts_dict.keys())
        GsizeDict = dict(zip(keys, GsizeList))
        # print(f"Minimum Area: {sol['cost']:.2f}")
        # print(GsizeList)
        return sol['cost'], GsizeDict
    except Exception as e:
        print("Solver failed:", e)
        return False, []


with open(sys.argv[1], 'r') as f: file_contents = f.read()  #sys.argv[1]
tspec = 1.3*twall.get(sys.argv[1])
circuit = parse(file_contents)
inpGates = gates_with_primary_inputs(circuit)
all_paths = circuit.get_all_paths()
fanouts_dict, critical_paths_dict = analyze_circuit(circuit)

import math
x = gates_with_all_primary_inputs(circuit)
output_gates = gates_with_primary_outputs(circuit)
N = len(fanouts_dict)
g = [1.0] * N
p = [2.0] * N

gate_index_map = {}
for gate in fanouts_dict:
    gate_type, num_inputs, _ = get_gate_type_and_inputs(gate)
    idx = int(gate.split("_")[1]) - 1
    gate_index_map[gate] = idx
    g[idx] = logicalEffortDict.get((gate_type, num_inputs), 1.0)
    p[idx] = num_inputs  

class Node:
    def __init__(self, name):
        self._name = name
        self._right = None
        self._left = None
        self._sol = None
        self._area = 0
        self.newconstraint = []

solList = []

eps = 1e-2
def is_sol_integer(sol, Nvar):
    for i in sol:
        if abs(round(i) - i) > eps:
            return False
    return True

class BnBNode:
    def __init__(self, constraints=None, depth=0, parent=None):
        self.constraints = constraints or []
        self.depth = depth
        self.parent = parent
        self.left_child = None
        self.right_child = None
        self.area = float('inf')
        self.sizes = {}
        self.is_feasible = False
        self.is_pruned = False
        self.prune_reason = None

import math
import copy
import time

time_limit = min(10*len(fanouts_dict), 1800)
print(time_limit)
last_area = math.inf 
best_sol = {}
def branch_and_bound(fanouts_dict, Tspec, inpGates, x, time_limit):
    start_time = time.time()
    best_area = float('inf')
    best_sizes = None
    
    # Track the best solution at each depth
    best_solution_by_depth = {}
    max_depth_reached = 0
    
    # Create root node
    root = BnBNode()
    
    # Solve the root node problem
    area, sizes = minimum_area_node_based(fanouts_dict, Tspec, inpGates, x, root.constraints)
    
    if area is False:
        root.is_feasible = False
        root.prune_reason = "Infeasible"
        return best_area, best_sizes
    
    root.area = area
    root.sizes = sizes
    root.is_feasible = True
    
    # Initialize best solution with root solution
    best_solution_by_depth[0] = (area, sizes)
    
    # Check if root solution is integer
    all_integer = all(abs(size - round(size)) < 1e-1 for size in sizes.values())
    if all_integer:
        return area, sizes
    
    # Initialize the queue with the root node
    queue = [root]
    
    while queue and (time.time() - start_time < time_limit):
        # Get the next node to process
        current_node = queue.pop(0)
        
        # Update max depth reached
        max_depth_reached = max(max_depth_reached, current_node.depth)
        
        # Skip if already pruned
        if current_node.is_pruned:
            continue
        
        # If node's lower bound exceeds best solution, prune it
        if current_node.area >= best_area:
            current_node.is_pruned = True
            current_node.prune_reason = "Bound exceeds best solution"
            continue
        
        # Track best solution at this depth
        if current_node.depth not in best_solution_by_depth or current_node.area < best_solution_by_depth[current_node.depth][0]:
            best_solution_by_depth[current_node.depth] = (current_node.area, current_node.sizes.copy())
        
        # If solution is integer, update best solution if better
        all_integer = all(abs(size - round(size)) < 1e-1 for size in current_node.sizes.values())
        if all_integer:
            if current_node.area < best_area:
                best_area = current_node.area
                best_sizes = current_node.sizes.copy()
            continue
        
        # Find a fractional variable to branch on
        branch_gate = None
        for gate, size in current_node.sizes.items():
            if abs(size - round(size)) > 1e-1:
                branch_gate = gate
                break
        
        if branch_gate is None:
            continue  # No fractional variables found
        
        floor_val = math.floor(current_node.sizes[branch_gate])
        ceil_val = math.ceil(current_node.sizes[branch_gate])
        
        # Create left child: gate ≤ floor_val
        left_child = BnBNode(
            constraints=current_node.constraints + [(branch_gate, '<=', floor_val)],
            depth=current_node.depth + 1,
            parent=current_node
        )
        
        current_node.left_child = left_child
        
        # Solve left child
        left_area, left_sizes = minimum_area_node_based(
            fanouts_dict, Tspec, inpGates, x, left_child.constraints
        )
        
        if left_area is not False:
            left_child.area = left_area
            left_child.sizes = left_sizes
            left_child.is_feasible = True
            
            # Prune if bound exceeds best solution
            if left_child.area >= best_area:
                left_child.is_pruned = True
                left_child.prune_reason = "Bound exceeds best solution"
            else:
                queue.append(left_child)
                
                # Update best solution at this depth
                depth = left_child.depth
                if depth not in best_solution_by_depth or left_area < best_solution_by_depth[depth][0]:
                    best_solution_by_depth[depth] = (left_area, left_sizes.copy())
                
                # Update best solution if integer
                if all(abs(size - round(size)) < 1e-1 for size in left_sizes.values()):
                    if left_area < best_area:
                        best_area = left_area
                        best_sizes = left_sizes.copy()
        else:
            left_child.is_feasible = False
            left_child.is_pruned = True
            left_child.prune_reason = "Infeasible"
        
        # Create right child: gate ≥ ceil_val
        right_child = BnBNode(
            constraints=current_node.constraints + [(branch_gate, '>=', ceil_val)],
            depth=current_node.depth + 1,
            parent=current_node
        )
        
        current_node.right_child = right_child
        
        # Solve right child
        right_area, right_sizes = minimum_area_node_based(
            fanouts_dict, Tspec, inpGates, x, right_child.constraints
        )
        
        if right_area is not False:
            right_child.area = right_area
            right_child.sizes = right_sizes
            right_child.is_feasible = True
            
            # Prune if bound exceeds best solution
            if right_child.area >= best_area:
                right_child.is_pruned = True
                right_child.prune_reason = "Bound exceeds best solution"
            else:
                queue.append(right_child)
                
                # Update best solution at this depth
                depth = right_child.depth
                if depth not in best_solution_by_depth or right_area < best_solution_by_depth[depth][0]:
                    best_solution_by_depth[depth] = (right_area, right_sizes.copy())
                
                # Update best solution if integer
                if all(abs(size - round(size)) < 1e-1 for size in right_sizes.values()):
                    if right_area < best_area:
                        best_area = right_area
                        best_sizes = right_sizes.copy()
        else:
            right_child.is_feasible = False
            right_child.is_pruned = True
            right_child.prune_reason = "Infeasible"
    
        print(f"Depth {current_node.depth}: Branched on {branch_gate} with value {current_node.sizes[branch_gate]}")
        print(f"  Left ({floor_val}): {'Feasible' if left_child.is_feasible else 'Infeasible'}, Area: {left_child.area if left_child.is_feasible else 'N/A'}")
        print(f"  Right ({ceil_val}): {'Feasible' if right_child.is_feasible else 'Infeasible'}, Area: {right_child.area if right_child.is_feasible else 'N/A'}")
    
    # Check if time limit was reached
    if time.time() - start_time >= time_limit:
        print(f"Time limit of {time_limit} seconds reached.")
        
        # If no integer solution found, use the best solution from the deepest level
        if best_area == float('inf') and max_depth_reached in best_solution_by_depth:
            best_area, best_sizes = best_solution_by_depth[max_depth_reached]
            print(f"Using best solution from max depth {max_depth_reached} with area {best_area}")
    
    # If no integer solution found, use the best solution from the deepest level
    if best_area == float('inf') and best_solution_by_depth:
        max_depth = max(best_solution_by_depth.keys())
        best_area, best_sizes = best_solution_by_depth[max_depth]
        print(f"No integer solution found. Using best solution from depth {max_depth} with area {best_area}")
    
    # Round the final solution to integers if needed
    if best_sizes:
        for name in best_sizes:
            i = best_sizes[name]
            if abs(i - round(i)) > 1e-1:
                if name in x:  # x contains primary input gates
                    best_sizes[name] = math.floor(i)
                else:
                    best_sizes[name] = math.ceil(i)
            else:
                best_sizes[name] = round(i)
        
        # Recalculate the final area
        best_area = sum(best_sizes.values())
    
    return best_area, best_sizes


start_time = time.time()
print(time)
a, b = branch_and_bound(fanouts_dict ,tspec, inpGates, x, time_limit)
print(a,b)
if b:
    best_sizes = b
    best_area = a
else:
    best_sizes = best_sol
    best_area = last_area
for name in best_sizes:
    i = best_sizes[name]
    if abs(i - round(i)) > 1e-1:
        if name in x:
            best_sizes[name] = math.floor(i)
        elif name in output_gates:
            best_sizes[name] = math.ceil(i)
        else:
            best_sizes[name] = round(i)
    else:
        best_sizes[name] = round(i)
best_area = sum(i for i in best_sizes.values())
print("Best area:", best_area)
print("Best sizes:", best_sizes)

import cvxpy as cp

# def solve_mixed_integer_gp(fanouts_dict, Tspec, inpGates, x):
#     N = len(fanouts_dict)
#     Gsize = cp.Variable(N, integer=True)  # Gate sizes as integer variables
#     T = cp.Variable(N)

#     g = [1.0] * N
#     p = [2.0] * N
#     gate_index_map = {}

#     for gate in fanouts_dict:
#         gate_type, num_inputs, _ = get_gate_type_and_inputs(gate)
#         idx = int(gate.split("_")[1]) - 1
#         gate_index_map[gate] = idx
#         g[idx] = logicalEffortDict.get((gate_type, num_inputs), 1.0)
#         p[idx] = num_inputs

#     constraints = [Gsize >= 1, Gsize <= max_gate_size]

#     # Setup delay constraints
#     for gate in fanouts_dict:
#         idx = gate_index_map[gate]
#         input_gates = circuit.gates[gate].outputs

#         # Fanout delay
#         total_fo_effort = 0
#         for fanout in fanouts_dict[gate]:
#             if fanout in gate_index_map:
#                 f_idx = gate_index_map[fanout]
#                 total_fo_effort += g[f_idx] * Gsize[f_idx]

#         delay = p[idx]
#         delay += total_fo_effort / Gsize[idx]

#         for input_gate in input_gates:
#             if input_gate.name in gate_index_map:
#                 input_idx = gate_index_map[input_gate.name]
#                 constraints.append(T[input_idx] >= T[idx] + delay)

#     # Primary input timing = 0
#     for gate_name in inpGates:
#         if gate_name in gate_index_map:
#             constraints.append(T[gate_index_map[gate_name]] == 0)

#     # Output gates must meet Tspec
#     for gate in gates_with_primary_outputs(circuit):
#         if gate in gate_index_map:
#             idx = gate_index_map[gate]
#             load_delay = cload / Gsize[idx]
#             constraints.append(T[idx] + load_delay + p[idx] <= Tspec)

#     # Objective: minimize area (sum of gate sizes)
#     objective = cp.Minimize(cp.sum(Gsize))

#     prob = cp.Problem(objective, constraints)
#     try:
#         prob.solve(solver=cp.GLPK_MI, verbose=True)  # GLPK_MI supports integer programming
#         GsizeList = [round(val) for val in Gsize.value]
#         keys = list(fanouts_dict.keys())
#         GsizeDict = dict(zip(keys, GsizeList))
#         print(f"MIGP Area: {sum(GsizeList)}")
#         print("MIGP Sizes:", GsizeDict)
#         return sum(GsizeList), GsizeDict
#     except Exception as e:
#         print("MIGP solver failed:", e)
#         return None, None

# def solve_mixed_integer_gp(fanouts_dict, Tspec, inpGates, x):
#     N = len(fanouts_dict)
#     Gsize = cp.Variable(N, integer=True)  # Gate sizes as integer variables
#     T = cp.Variable(N)  # Arrival times
    
#     g = [1.0] * N
#     p = [2.0] * N
#     gate_index_map = {}
    
#     # Map gates to indices
#     for i, gate in enumerate(fanouts_dict.keys()):
#         gate_type, num_inputs, _ = get_gate_type_and_inputs(gate)
#         gate_index_map[gate] = i
#         g[i] = logicalEffortDict.get((gate_type, num_inputs), 1.0)
#         p[i] = num_inputs
    
#     constraints = [Gsize >= 1, Gsize <= max_gate_size]
    
#     # Setup delay constraints using linear approximation
#     for gate in fanouts_dict:
#         idx = gate_index_map[gate]
#         input_gates = circuit.gates[gate].outputs
        
#         # For each input gate, create a timing constraint
#         for input_gate in input_gates:
#             if input_gate.name in gate_index_map:
#                 input_idx = gate_index_map[input_gate.name]
                
#                 # Instead of division, use multiplication on the other side
#                 # Original: T[input_idx] >= T[idx] + p[idx] + total_fo_effort / Gsize[idx]
#                 # Rewrite as: T[input_idx] * Gsize[idx] >= T[idx] * Gsize[idx] + p[idx] * Gsize[idx] + total_fo_effort
                
#                 # But this is still not linear. So we'll use a piecewise linear approximation
#                 # or a different approach:
                
#                 # Approach 1: Use a conservative fixed delay estimate based on minimum gate size
#                 min_delay = p[idx] + (sum(g[gate_index_map[fo]] * max_gate_size for fo in fanouts_dict[gate])) / 1.0
#                 constraints.append(T[input_idx] >= T[idx] + min_delay)
                
#                 # Approach 2: Create multiple constraints for different possible gate sizes
#                 # This is more accurate but adds more constraints
#                 for size in range(1, max_gate_size + 1):
#                     # If Gsize[idx] == size, then this constraint should apply
#                     # We can model this with indicator constraints if the solver supports them
#                     total_fo_effort = sum(g[gate_index_map[fo]] * max_gate_size for fo in fanouts_dict[gate])
#                     delay_at_size = p[idx] + total_fo_effort / size
                    
#                     # Create indicator constraint: Gsize[idx] == size => T[input_idx] >= T[idx] + delay_at_size
#                     # Many solvers support this directly, but for GLPK_MI we need to reformulate:
#                     M = 1000  # A large constant
#                     indicator = cp.Variable(boolean=True)
#                     constraints.append(Gsize[idx] <= size - 1 + M * indicator)
#                     constraints.append(Gsize[idx] >= size - M * (1 - indicator))
#                     constraints.append(T[input_idx] >= T[idx] + delay_at_size - M * (1 - indicator))
    
#     # Primary input timing = 0
#     for gate_name in inpGates:
#         if gate_name in gate_index_map:
#             constraints.append(T[gate_index_map[gate_name]] == 0)
    
#     # Output gates must meet Tspec
#     for gate in gates_with_primary_outputs(circuit):
#         if gate in gate_index_map:
#             idx = gate_index_map[gate]
#             # Again, avoid division
#             for size in range(1, max_gate_size + 1):
#                 load_delay = cload / size
                
#                 # Create indicator constraint
#                 M = 1000
#                 indicator = cp.Variable(boolean=True)
#                 constraints.append(Gsize[idx] <= size - 1 + M * indicator)
#                 constraints.append(Gsize[idx] >= size - M * (1 - indicator))
#                 constraints.append(T[idx] + load_delay + p[idx] <= Tspec + M * (1 - indicator))
    
#     # Objective: minimize area (sum of gate sizes)
#     objective = cp.Minimize(cp.sum(Gsize))
    
#     prob = cp.Problem(objective, constraints)
#     try:
#         # Use a solver that supports integer and binary variables
#         prob.solve(solver=cp.GLPK_MI, verbose=True)
        
#         if prob.status == cp.OPTIMAL:
#             GsizeList = [int(round(val)) for val in Gsize.value]
#             keys = list(fanouts_dict.keys())
#             GsizeDict = dict(zip(keys, GsizeList))
#             print(f"MIGP Area: {sum(GsizeList)}")
#             print("MIGP Sizes:", GsizeDict)
#             return sum(GsizeList), GsizeDict
#         else:
#             print(f"Solver status: {prob.status}")
#             return None, None
#     except Exception as e:
#         print("MIGP solver failed:", e)
#         return None, None


# solve_mixed_integer_gp(fanouts_dict, tspec, inpGates, x)