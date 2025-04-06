import math
from math import radians

import bpy
import bmesh
import mathutils

import os
import numpy as np
from scipy.spatial.transform import Rotation as R

os.system('cls')


# _______________________________________________________________________________________________________________________________________
# LINE DRAWING COMMANDS
# _______________________________________________________________________________________________________________________________________
# F = Move forward a step, drawing a line between previous pos and new pos
# H = Move forward half a step, drawing a line between previous pos and new pos

# Without additional info, these commands use 'Step Length' as length value

# F(l) = Draws a line of length 'l'
# F(l,w) = Draws a line of length 'l' and width 'w'

# _______________________________________________________________________________________________________________________________________
# ROTATION COMMANDS
# _______________________________________________________________________________________________________________________________________
# + = Turn right default 'Angle' degrees
# - = Turn left default 'Angle' degrees
# & = Pitch up default 'Angle' degrees
# ^ = Pitch down default 'Angle' degrees
# \\ = Roll clockwise default 'Angle' degrees
# / = Roll counter-clockwise default 'Angle' degrees
# | = Turn 180 degrees
# * = Roll 180 degrees
# ~ = Pitch / Roll / Turn random amount up to default 'Angle' degrees

# Without additional info, these commands use 'Angle' as the value to determine the angle rotated in each turn

# +(a)
# &(a)
# ect. = The 'a' value is used to set the angle instead of the default 'Angle' variable

# _______________________________________________________________________________________________________________________________________
# TAKE THE PREMISE AND USE THE RULES TO BUILD THE STRING THE L-SYSTEM IS DRAWN FROM
# _______________________________________________________________________________________________________________________________________

# _______________________________________________________________________________________________________________________________________
# FUNCTIONS NECESSARY FOR L-SYSTEM DRAW FUNCTION
# _______________________________________________________________________________________________________________________________________


def initial_lw_value(s, d):
    """Calculate the initial value given to l and w in an L-System"""
    index = 0
    for char in s:
        index += 1
        if index < len(s):
            if s[index] != '(':
                continue
            else:
                for key, value in d.items():
                    if char == key[0]:
                        s_chopped = s[index:]
                        s_chopped_index = 0

                        for ch in s_chopped:
                            match ch:
                                case ')' | ',':
                                    break
                                case _:
                                    s_chopped_index += 1

                        l = s_chopped[1:s_chopped_index]
                        w = s_chopped[s_chopped_index + 1:-1]
                        return float(l), float(w)
                    else:
                        continue

        else:
            return 1, 1


def branch_count(s, d, c):
    """Count the total number of branches generated in current generation"""
    nr = 0
    if len(d) > 0:
        for key, value in d.items():
            if key == "":
                continue
            else:
                if key[0] in s:
                    nr += s.count(key[0])
                else:
                    continue
        c *= nr
        return c
    else:
        return 1


# _______________________________________________________________________________________________________________________________________
# UTILITY FUNCTIONS FOR ADDITIONAL RULES
# _______________________________________________________________________________________________________________________________________


def update_value(s, l, itera, current_iter, stack, b, c, d, q, p, r):
    """Update the value of l depending on witch generation is currently being generated"""
    if current_iter >= itera - 1:
        return l
    l = eval(s)
    stack.append(l)
    update_value(s, l, itera, current_iter + 1, stack, b, c, d, q, p, r)
    l = stack.pop(0)
    print(f'final l: {l}')
    return l


def value_override(input_string, index, initial_value, l, b, c, d, q, p, r, iters):
    """Calculate the override value given later in the input string if set"""
    if index >= len(input_string):
        return initial_value
    else:
        if input_string[index] != '(':
            return initial_value

        else:
            match input_string[index - 1]:
                case "+" | "-" | "&" | "^" | "\\" | "/" | "~":
                    print(input_string[index - 1])

                    chopped_input_string = input_string[index:]
                    chopped_input_string_index = 0

                    for char in chopped_input_string:
                        match char:
                            case ')' | ',':
                                break
                            case _:
                                chopped_input_string_index += 1

                    override_string = chopped_input_string[1:chopped_input_string_index]
                    if 'l' in override_string:
                        l_stack = []
                        current_iter = 0
                        l = update_value(override_string, l, iters, current_iter, l_stack, b, c, d, q, p, r)
                        return radians(l)
                    print(f'override value for rotation: {eval(override_string)}')
                    override_value = radians(eval(override_string))
                    return override_value

                case _:
                    chopped_input_string = input_string[index:]
                    chopped_input_string_index = 0

                    for char in chopped_input_string:
                        match char:
                            case ')' | ',':
                                break
                            case _:
                                chopped_input_string_index += 1

                    override_string = chopped_input_string[1:chopped_input_string_index]
                    if 'l' in override_string:
                        l_stack = []
                        current_iter = 0
                        l = update_value(override_string, l, iters, current_iter, l_stack, b, c, d, q, p, r)
                        print(f'current l after function: {l}')
                        return l
                    override_value = eval(override_string)
                    print(f'override step length: {override_value}')
                    return override_value


def width_override(input_string, index, w, b, c, d, q, p, r):
    """Calculate what the desired width scaling factor is for current step"""
    if index >= len(input_string):
        return 1.0
    else:
        if input_string[index] != '(':
            return 1.0
        else:
            chopped_input_string = input_string[index:]
            chopped_input_string_index = 0
            for char in chopped_input_string:
                match char:
                    case ')':
                        break
                    case _:
                        chopped_input_string_index += 1
            override_string = chopped_input_string[1:chopped_input_string_index]
            w_string = ""
            chopped_input_string_index = 0
            for char in override_string:
                match char:
                    case ',':
                        w_string = override_string[chopped_input_string_index + 1:]
                        break
                    case _:
                        chopped_input_string_index += 1
            chopped_input_string_index = 0
            if len(w_string) > 0 and '*' in w_string:
                for char in w_string:
                    match char:
                        case '*':
                            chopped_input_string_index += 1
                            w_string = w_string[chopped_input_string_index:]
                            break
                        case _:
                            chopped_input_string_index += 1
                return float(eval(w_string))
            else:
                return 1.0


def smallest_rotation(v1, v2):
    """Compute the smallest rotation axis and angle needed to rotate v1 to v2."""

    v1 = mathutils.Vector(v1).normalized()
    v2 = mathutils.Vector(v2).normalized()

    axis_of_rotation = v1.cross(v2)

    dot_pro = max(-1.0, min(1.0, v1.dot(v2)))
    rotation_angle = math.acos(dot_pro)

    if axis_of_rotation < 1e-6:
        return None, 0.0

    return axis_of_rotation.normalized(), rotation_angle


def update_shape(self, context):
    """Function triggered when shape changes"""
    if self.shape_toggle == 'TUBE':
        self.use_tube = True
    else:
        self.use_tube = False


# _______________________________________________________________________________________________________________________________________
# DRAW FUNCTION
# _______________________________________________________________________________________________________________________________________


def draw_l_system(rules, current_string, l_sys_angle, step_size, old_position, max_iters, current_iter,
                  current_rotation,
                  current_vert_index, current_edge_index, branch_amount, tube_res, tube_radi, l, w,
                  custom_b, custom_c, custom_d, custom_q, custom_p, custom_r, i_stack, b_stack, tube):
    """Generate L-System with the given rules and conditions"""

    new_position = old_position
    for key, value in rules.items():
        print(key, value)
    print(f'current iteration: {current_iter}')
    print(f'l currently equals: {l}')
    if current_iter >= max_iters:
        print('break')
        return

    if current_iter == 0:

        if tube:
            bpy.ops.mesh.primitive_circle_add(vertices=tube_res, radius=tube_radi, location=old_position)
            bpy.context.object.name = "L-System"
            bpy.ops.object.mode_set(mode='EDIT')

            bpy.ops.mesh.select_all(
                action='DESELECT')  # deselect everything first so we don't select any stray vertices or edges

            # create a bmesh from the current selection to be able to directly manipulate the vertices
            obj = bpy.context.object
            bm = bmesh.from_edit_mesh(obj.data)
            bm.edges.ensure_lookup_table()
            # indices_to_select = 3  # Select the first edges
            for i in range(tube_res):
                bm.edges.ensure_lookup_table()
                current_edge = bm.edges[i]
                current_edge_index.append(current_edge.index)
                bm.edges[i].select = True
            bmesh.update_edit_mesh(obj.data)

        else:
            # Add a circle mesh to make a single point at origin
            bpy.ops.mesh.primitive_circle_add(vertices=3, radius=0, location=old_position)
            bpy.context.object.name = "L-System"
            bpy.ops.object.mode_set(mode='EDIT')

            bpy.ops.mesh.select_all(
                action='DESELECT')  # deselect everything first so we don't select any stray vertices or edges

            # create a bmesh from the current selection to be able to directly manipulate the vertices
            obj = bpy.context.object
            bm = bmesh.from_edit_mesh(obj.data)
            bm.verts.ensure_lookup_table()
            bm.verts[0].select = True  # select the first vertex
            bmesh.update_edit_mesh(obj.data)

    else:
        obj = bpy.context.object
        bm = bmesh.from_edit_mesh(obj.data)

    for i in range(branch_amount):
        print(f'i={i} max= {branch_amount}, while iteration={current_iter}')
        print(f'current rotation: {current_rotation.as_euler("XYZ", degrees=True)}')
        # Set the string index of the first character
        current_string_index = 0

        for character in current_string:  # loop through each character in end string to draw the l-system
            print(f'character in match case: {character}')

            # Increment the string index for each character being looped through
            current_string_index += 1

            match character:

                case 'F':

                    previous_position = new_position
                    forward_vector = np.array([0, 0, value_override(current_string, current_string_index, step_size, l,
                                                                    custom_b, custom_c, custom_d,
                                                                    custom_q, custom_p, custom_r, current_iter)])

                    # rotate the extruded point to get its rotated coordinates
                    new_position = current_rotation.apply(forward_vector)

                    # select the saved vertex to ensure the extrusion happens from where we want
                    bpy.ops.mesh.select_all(
                        action='DESELECT')  # deselect everything first so we don't select any stray vertices or edges
                    if tube:
                        print(current_edge_index)
                        for k in current_edge_index:
                            bm.edges.ensure_lookup_table()
                            bm.edges[k].select = True
                            bmesh.update_edit_mesh(obj.data)

                        bpy.ops.mesh.extrude_edges_move(
                            TRANSFORM_OT_translate={"value": new_position})  # extrude edges to the new position

                        bm = bmesh.from_edit_mesh(obj.data)
                        bm.edges.ensure_lookup_table()

                        current_edge_index.clear()
                        for k in range(tube_res):
                            bm.edges.ensure_lookup_table()
                            current_edge = bm.edges[-k - (tube_res + 1)]
                            current_edge_index.append(current_edge.index)

                        edge_verts = set()
                        for edge in current_edge_index:
                            edge_verts.update(bm.edges[edge].verts)

                        # Get center position of edge loop
                        center = sum((v.co for v in edge_verts), mathutils.Vector()) / len(edge_verts)

                        print(f'previous direction= {previous_position}')
                        print(f'new direction= {new_position}')

                        axis, angle = smallest_rotation(previous_position, new_position)
                        if angle > 1e-6:
                            rotation_matrix = mathutils.Matrix.Rotation(angle, 3, axis)
                            for v in edge_verts:
                                v.co = center + rotation_matrix @ (v.co - center)

                        bmesh.update_edit_mesh(obj.data)
                        bpy.ops.transform.resize(
                            value=(width_override(current_string, current_string_index, w, custom_b,
                                                  custom_c, custom_d, custom_q, custom_p, custom_r),
                                   width_override(current_string, current_string_index, w, custom_b,
                                                  custom_c, custom_d, custom_q, custom_p, custom_r),
                                   width_override(current_string, current_string_index, w, custom_b,
                                                  custom_c, custom_d, custom_q, custom_p, custom_r)),
                            orient_matrix=((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)))
                        bmesh.update_edit_mesh(obj.data)
                        bm = bmesh.from_edit_mesh(obj.data)

                    else:
                        bm.verts.ensure_lookup_table()
                        bm.verts[current_vert_index].select = True  # select the saved vertex
                        bmesh.update_edit_mesh(obj.data)

                        bpy.ops.mesh.extrude_vertices_move(
                            TRANSFORM_OT_translate={"value": new_position})  # extrude vertex to the new position

                        bm = bmesh.from_edit_mesh(obj.data)
                        bm.verts.ensure_lookup_table()

                        current_vert = bm.verts[-1]  # get the index of the last vertex
                        current_vert_index = current_vert.index  # update the current vertex index

                case 'H':

                    previous_position = new_position
                    half_forward_vector = np.array([0, 0, step_size / 2])
                    new_position = current_rotation.apply(half_forward_vector)

                    # select the saved vertex to ensure the extrusion happens from where we want
                    bpy.ops.mesh.select_all(
                        action='DESELECT')  # deselect everything first so we don't select any stray vertices or edges
                    if tube:
                        for k in current_edge_index:
                            bm.edges.ensure_lookup_table()
                            bm.edges[k].select = True
                            bmesh.update_edit_mesh(obj.data)

                        bpy.ops.mesh.extrude_edges_move(
                            TRANSFORM_OT_translate={"value": new_position})  # extrude edges to the new position

                        bm = bmesh.from_edit_mesh(obj.data)
                        bm.edges.ensure_lookup_table()

                        current_edge_index.clear()
                        for k in range(tube_res):
                            bm.edges.ensure_lookup_table()
                            current_edge = bm.edges[-k - (tube_res + 1)]
                            current_edge_index.append(current_edge.index)

                        edge_verts = set()
                        for edge in current_edge_index:
                            edge_verts.update(bm.edges[edge].verts)

                        # Get center position of edge loop
                        center = sum((v.co for v in edge_verts), mathutils.Vector()) / len(edge_verts)

                        print(f'previous direction= {previous_position}')
                        print(f'new direction= {new_position}')

                        axis, angle = smallest_rotation(previous_position, new_position)
                        if angle > 1e-6:
                            rotation_matrix = mathutils.Matrix.Rotation(angle, 3, axis)
                            for v in edge_verts:
                                v.co = center + rotation_matrix @ (v.co - center)

                        bmesh.update_edit_mesh(obj.data)
                        bpy.ops.transform.resize(value=(
                            width_override(current_string, current_string_index, w, custom_b,
                                           custom_c, custom_d, custom_q, custom_p, custom_r),
                            width_override(current_string, current_string_index, w, custom_b,
                                           custom_c, custom_d, custom_q, custom_p, custom_r),
                            width_override(current_string, current_string_index, w, custom_b,
                                           custom_c, custom_d, custom_q, custom_p, custom_r)),
                            orient_matrix=((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)))
                        bmesh.update_edit_mesh(obj.data)
                        bm = bmesh.from_edit_mesh(obj.data)

                    else:
                        bm.verts.ensure_lookup_table()
                        bm.verts[current_vert_index].select = True  # select the saved vertex
                        bmesh.update_edit_mesh(obj.data)

                        bpy.ops.mesh.extrude_vertices_move(
                            TRANSFORM_OT_translate={"value": new_position})  # extrude vertex to the new position

                        bm = bmesh.from_edit_mesh(obj.data)
                        bm.verts.ensure_lookup_table()

                        current_vert = bm.verts[-1]  # get the index of the last vertex
                        current_vert_index = current_vert.index  # update the current vertex index

                case '+':
                    yaw_rotation = R.from_euler('y', -value_override(current_string, current_string_index,
                                                                     l_sys_angle, l,
                                                                     custom_b, custom_c,
                                                                     custom_d, custom_q,
                                                                     custom_p, custom_r,
                                                                     current_iter))
                    current_rotation = current_rotation * yaw_rotation

                case '-':
                    yaw_rotation = R.from_euler('y', value_override(current_string, current_string_index,
                                                                     l_sys_angle, l,
                                                                     custom_b, custom_c,
                                                                     custom_d, custom_q,
                                                                     custom_p, custom_r,
                                                                     current_iter))
                    current_rotation = current_rotation * yaw_rotation

                case '&':
                    pitch_rotation = R.from_euler('x', -value_override(current_string, current_string_index,
                                                                     l_sys_angle, l,
                                                                     custom_b, custom_c,
                                                                     custom_d, custom_q,
                                                                     custom_p, custom_r,
                                                                     current_iter))
                    current_rotation = current_rotation * pitch_rotation

                case '^':
                    pitch_rotation = R.from_euler('x', value_override(current_string, current_string_index,
                                                                     l_sys_angle, l,
                                                                     custom_b, custom_c,
                                                                     custom_d, custom_q,
                                                                     custom_p, custom_r,
                                                                     current_iter))
                    current_rotation = current_rotation * pitch_rotation

                case '\\':
                    roll_rotation = R.from_euler('z', -value_override(current_string, current_string_index,
                                                                     l_sys_angle, l,
                                                                     custom_b, custom_c,
                                                                     custom_d, custom_q,
                                                                     custom_p, custom_r,
                                                                     current_iter))
                    current_rotation = current_rotation * roll_rotation

                case '/':
                    roll_rotation = R.from_euler('z', value_override(current_string, current_string_index,
                                                                     l_sys_angle, l,
                                                                     custom_b, custom_c,
                                                                     custom_d, custom_q,
                                                                     custom_p, custom_r,
                                                                     current_iter))
                    print(f'pre multiplication: {current_rotation.as_euler("XYZ", degrees=True)}')
                    print(f'rotation to multiply with: {roll_rotation.as_euler("XYZ", degrees=True)}')
                    current_rotation = current_rotation * roll_rotation
                    print(f'after multiplication: {current_rotation.as_euler("XYZ", degrees=True)}')

                case '|':
                    yaw_rotation = R.from_euler('y', 180, degrees=True)
                    current_rotation = current_rotation * yaw_rotation

                case '*':
                    roll_rotation = R.from_euler('z', 180, degrees=True)
                    current_rotation = current_rotation * roll_rotation

                case '[':

                    b_stack.append((current_rotation.as_quat().copy(), current_vert_index,
                                    current_edge_index.copy(), new_position))

                case ']':
                    saved_rotation_quat, current_vert_index, current_edge_index, new_position = b_stack.pop()
                    current_rotation = R.from_quat(saved_rotation_quat)

                    if tube:
                        bpy.ops.mesh.select_all(
                            action='DESELECT')  # deselect everything first so we don't select any stray vertices or edges
                        for k in current_edge_index:
                            bm.edges.ensure_lookup_table()
                            bm.edges[k].select = True
                            bmesh.update_edit_mesh(obj.data)

                    else:
                        # select the saved vertex to ensure the extrusion happens from where we want
                        bpy.ops.mesh.select_all(
                            action='DESELECT')  # deselect everything first so we don't select any stray vertices or edges
                        bm.verts.ensure_lookup_table()
                        bm.verts[current_vert_index].select = True  # select the saved vertex
                        bmesh.update_edit_mesh(obj.data)

                case '~':
                    randomize_angle_yaw = np.random.uniform(
                        -value_override(current_string, current_string_index,
                                        l_sys_angle, l,
                                        custom_b, custom_c,
                                        custom_d, custom_q,
                                        custom_p, custom_r,
                                        current_iter),
                        value_override(current_string, current_string_index,
                                        l_sys_angle, l,
                                        custom_b, custom_c,
                                        custom_d, custom_q,
                                        custom_p, custom_r,
                                        current_iter))
                    randomize_angle_pitch = np.random.uniform(
                        -value_override(current_string, current_string_index,
                                        l_sys_angle, l,
                                        custom_b, custom_c,
                                        custom_d, custom_q,
                                        custom_p, custom_r,
                                        current_iter),
                        value_override(current_string, current_string_index,
                                       l_sys_angle, l,
                                       custom_b, custom_c,
                                       custom_d, custom_q,
                                       custom_p, custom_r,
                                       current_iter))
                    randomize_angle_roll = np.random.uniform(
                        -value_override(current_string, current_string_index,
                                        l_sys_angle, l,
                                        custom_b, custom_c,
                                        custom_d, custom_q,
                                        custom_p, custom_r,
                                        current_iter),
                        value_override(current_string, current_string_index,
                                       l_sys_angle, l,
                                       custom_b, custom_c,
                                       custom_d, custom_q,
                                       custom_p, custom_r,
                                       current_iter))

                    yaw_rotation = R.from_euler('y', randomize_angle_yaw)
                    current_rotation = current_rotation * yaw_rotation

                    pitch_rotation = R.from_euler('x', randomize_angle_pitch)
                    current_rotation = current_rotation * pitch_rotation

                    roll_rotation = R.from_euler('z', randomize_angle_roll)
                    current_rotation = current_rotation * roll_rotation

                case _:
                    pass

            for rule, result in rules.items():
                print(f'rule: {rule}')
                print(f'character: {character}')
                if rule == "":
                    continue
                else:
                    if character.isalpha() and character in rule[0]:
                        print(current_string)
                        next_branch_amount = branch_count(current_string, rules, branch_amount)
                        print(next_branch_amount)
                        overwritten_character = result
                        i_stack.append((overwritten_character, current_rotation.as_quat().copy(), current_vert_index,
                                        current_edge_index.copy(), next_branch_amount, new_position))
                        print(i_stack)

                    else:
                        print("Character not in rules")
                        continue

        # print(i_stack)
        if (i < branch_amount - 1) & (len(i_stack) > 0):
            print(i_stack)
            current_string, saved_rot, current_vert_index, current_edge_index, branch_amount, new_position = i_stack.pop(
                0)
            current_rotation = R.from_quat(saved_rot)
            print(f'post pop: {i_stack}')

    if len(i_stack) > 0:
        print(i_stack)
        current_string, saved_rot, current_vert_index, current_edge_index, branch_amount, new_position = i_stack.pop(0)
        current_rotation = R.from_quat(saved_rot)
        print(f'post pop: {i_stack}')

    print(
        f'current string: {current_string}, current rotation: {current_rotation.as_euler("XYZ")}, current edge index: {current_edge_index}')
    draw_l_system(rules, current_string, l_sys_angle, step_size, new_position, max_iters, current_iter + 1,
                  current_rotation,
                  current_vert_index, current_edge_index, branch_amount, tube_res, tube_radi, l, w,
                  custom_b, custom_c, custom_d, custom_q, custom_p, custom_r, i_stack, b_stack, tube)
    print(f'jumped to iteration={current_iter}')


# _______________________________________________________________________________________________________________________________________
# Classes for user input and creating panel UI
# _______________________________________________________________________________________________________________________________________


class LSystemProperties(bpy.types.PropertyGroup):
    """Create inputs to pass to the Draw L-System function"""

    premise: bpy.props.StringProperty(
        name="Premise",
        description="Enter the premise",
        default="FFFA"
    )

    rules_1: bpy.props.StringProperty(
        name="Rules 1",
        description="Enter rule in A=F format",
        default="B=&FFFA"
    )

    rules_2: bpy.props.StringProperty(
        name="Rules 2",
        description="Enter rule in A=F format",
        default="A=[B]////[B]////[B]"
    )

    rules_3: bpy.props.StringProperty(
        name="Rules 3",
        description="Enter rule in A=F format",
        default="="
    )

    rules_4: bpy.props.StringProperty(
        name="Rules 4",
        description="Enter rule in A=F format",
        default="="
    )

    l_sys_angle: bpy.props.FloatProperty(
        name="Angle",
        description="Choose rotation angle",
        default=radians(28),
        subtype="ANGLE"
    )

    iterations: bpy.props.IntProperty(
        name="Iterations",
        description="Number of iterations",
        default=8
    )

    step_length: bpy.props.FloatProperty(
        name="Step Length",
        description="The length of each step in meters",
        default=1.0
    )

    custom_b: bpy.props.FloatProperty(
        name="Custom Variable b",
        description="Custom variable to control values, such as step length overrides",
        default=0.0
    )

    custom_c: bpy.props.FloatProperty(
        name="Custom Variable c",
        description="Custom variable to control values, such as step length overrides",
        default=0.0
    )

    custom_d: bpy.props.FloatProperty(
        name="Custom Variable d",
        description="Custom variable to control values, such as step length overrides",
        default=0.0
    )

    custom_q: bpy.props.FloatProperty(
        name="Custom Variable q",
        description="Custom variable to control values, such as step length overrides",
        default=0.0
    )

    custom_p: bpy.props.FloatProperty(
        name="Custom Variable p",
        description="Custom variable to control values, such as step length overrides",
        default=0.0
    )

    custom_r: bpy.props.FloatProperty(
        name="Custom Variable r",
        description="Custom variable to control values, such as step length overrides",
        default=0.0
    )

    use_tube: bpy.props.BoolProperty(
        name="Shape",
        default=False
    )

    tube_resolution: bpy.props.IntProperty(
        name="Tube Resolution",
        description="Number of sides on the generated tube",
        default=6,
        min=0
    )

    tube_radius: bpy.props.FloatProperty(
        name="Tube Radius",
        description="Initial radius of generated tube",
        default=0.25,
        min=0.0
    )

    shape_toggle: bpy.props.EnumProperty(
        name="Shape",
        description="Draw L-System as wire or tube",
        items=[
            ('WIRE', "Wire", "Draw L-System as wire"),
            ('TUBE', "Tube", "Draw L-System as tube")
        ],
        default='WIRE',
        update=update_shape
    )

    preset_options: bpy.props.EnumProperty(
        name="Presets",
        description="Choose a preset",
        items=[
            ('SIMPLE_TREE', "Simple Tree", "Use rule preset for a simple tree"),
            ('MONO_TREE', "Monopodial Tree", "Use preset for a Monopodial tree"),
            ('DRAGON_CURVE', "Dragon Curve", "Use rule preset for the Dragon Curve"),
            ('SIERP_TRIANGLE', "Sierpinski Triangle", "Use rule preset for the Sierpinski Triangle"),
            ('KOCH_CURVE', "Koch Curve", "Use rule preset for a Koch Curve"),
            ('ROLL_CHECK', "Roll Checker", "Use rule preset to check roll is functioning")
        ],
        default='SIMPLE_TREE'
    )


class ApplyPresets(bpy.types.Operator):
    """Apply the chosen preset to the inputs"""

    bl_idname = "mesh.apply_presets"
    bl_label = "Apply Presets"

    def execute(self, context):
        scene = context.scene
        lsystem_props = scene.lsystem_props

        # Apply the selected preset
        if lsystem_props.preset_options == 'SIMPLE_TREE':
            lsystem_props.premise = "FFFA"
            lsystem_props.rules_1 = "B=&FFFA"
            lsystem_props.rules_2 = "A=[B]////[B]////[B]"
            lsystem_props.rules_3 = ""
            lsystem_props.rules_4 = ""
            lsystem_props.l_sys_angle = radians(28)
            lsystem_props.iterations = 8
            lsystem_props.step_length = 1.0

        elif lsystem_props.preset_options == 'MONO_TREE':
            lsystem_props.premise = "A(1,10)"
            lsystem_props.rules_1 = "A(l,w)=F(l*0.9,w*0.707)[&(c)B]/(137.5)A"
            lsystem_props.rules_2 = "B(l,w)=F(l*b,w*0.707)[-(d)C]C"
            lsystem_props.rules_3 = "C(l,w)=F(l*b,w*0.707)[+(d)B]B"
            lsystem_props.rules_4 = ""
            lsystem_props.l_sys_angle = radians(12)
            lsystem_props.iterations = 6
            lsystem_props.step_length = 0.05
            lsystem_props.custom_b = 0.8
            lsystem_props.custom_c = 35
            lsystem_props.custom_d = 45
            lsystem_props.shape_toggle = 'TUBE'
            lsystem_props.tube_resolution = 5
            lsystem_props.tube_radius = 0.1

        elif lsystem_props.preset_options == 'DRAGON_CURVE':
            lsystem_props.premise = "FX"
            lsystem_props.rules_1 = "X=X+YF+"
            lsystem_props.rules_2 = "Y=-FX-Y"
            lsystem_props.rules_3 = ""
            lsystem_props.rules_4 = ""
            lsystem_props.l_sys_angle = radians(90)
            lsystem_props.iterations = 6
            lsystem_props.step_length = 1.0

        elif lsystem_props.preset_options == 'SIERP_TRIANGLE':
            lsystem_props.premise = "A"
            lsystem_props.rules_1 = "A=+F-A-F+"
            lsystem_props.rules_2 = "F=-A+F+A-"
            lsystem_props.rules_3 = ""
            lsystem_props.rules_4 = ""
            lsystem_props.l_sys_angle = radians(60)
            lsystem_props.iterations = 8
            lsystem_props.step_length = 1.0

        elif lsystem_props.preset_options == 'KOCH_CURVE':
            lsystem_props.premise = "F-F-F-F"
            lsystem_props.rules_1 = "F=FF-F-F-F-FF"
            lsystem_props.rules_2 = ""
            lsystem_props.rules_3 = ""
            lsystem_props.rules_4 = ""
            lsystem_props.l_sys_angle = radians(90)
            lsystem_props.iterations = 4
            lsystem_props.step_length = 1.0

        elif lsystem_props.preset_options == 'ROLL_CHECK':
            lsystem_props.premise = "FA"
            lsystem_props.rules_1 = "A=A/[+F]"
            lsystem_props.rules_2 = ""
            lsystem_props.rules_3 = ""
            lsystem_props.rules_4 = ""
            lsystem_props.l_sys_angle = radians(30)
            lsystem_props.iterations = 8
            lsystem_props.step_length = 1.0

        return {"FINISHED"}


class CreateLSystem(bpy.types.Operator):
    """Generate L System from the given rules, premise, iteration and angle"""

    bl_idname = "mesh.create_l_sys"
    bl_label = "Create L-System"

    def execute(self, context):
        os.system('cls')
        scene = context.scene
        lsystem_props = scene.lsystem_props

        premise = lsystem_props.premise
        l_sys_angle = lsystem_props.l_sys_angle
        iterations = lsystem_props.iterations
        step_length = lsystem_props.step_length

        use_tube = lsystem_props.use_tube
        tube_resolution = lsystem_props.tube_resolution
        tube_radius = lsystem_props.tube_radius

        custom_b = lsystem_props.custom_b
        custom_c = lsystem_props.custom_c
        custom_d = lsystem_props.custom_d
        custom_q = lsystem_props.custom_q
        custom_p = lsystem_props.custom_p
        custom_r = lsystem_props.custom_r

        rule_1 = lsystem_props.rules_1
        rule_2 = lsystem_props.rules_2
        rule_3 = lsystem_props.rules_3
        rule_4 = lsystem_props.rules_4

        # make sure the rules dictionary exists in the scene
        rules = scene['rules'] = {}

        # Split all rule strings into its half and insert into a dictionary
        if ("=" in rule_1) and (len(rule_1) > 2):
            key_1, value_1 = rule_1.split("=", 1)  # Split string at first '=' sign

            rules[key_1.strip()] = value_1.strip()

        if ("=" in rule_2) and (len(rule_2) > 2):
            key_2, value_2 = rule_2.split("=", 1)

            rules[key_2.strip()] = value_2.strip()

        if ("=" in rule_3) and (len(rule_3) > 2):
            key_3, value_3 = rule_3.split("=", 1)

            rules[key_3.strip()] = value_3.strip()

        if ("=" in rule_4) and (len(rule_4) > 2):
            key_4, value_4 = rule_4.split("=", 1)

            rules[key_4.strip()] = value_4.strip()

        for key,value in rules.items():
            print(key,value)

        # stack to save current character, position and angle, so that iterations can be processed successively
        iteration_stack = []  # stack for saving information between iterations and branches
        current_iteration = 0  # internal tracker for current iteration
        current_branch_count = 1  # internal tracker for how many branches are generated each iteration
        branch_stack = []  # stack to save current position and angle, so that branches can be created
        start_position = 0, 0, 0
        # Initial rotation of steps
        start_rotation = R.from_quat([0, 0, 0, 1])
        # save the initial vertex index for the first point / create list for each set of edges
        first_vert_index = 0
        first_edge_indices = []

        print(initial_lw_value(premise, rules))

        initial_l, initial_w = initial_lw_value(premise, rules)
        b_count = branch_count(premise, rules, current_branch_count)

        draw_l_system(rules, premise, l_sys_angle, step_length, start_position, iterations, current_iteration,
                      start_rotation,
                      first_vert_index, first_edge_indices, b_count, tube_resolution, tube_radius, initial_l, initial_w,
                      custom_b, custom_c, custom_d, custom_q, custom_p, custom_r, iteration_stack, branch_stack,
                      use_tube)

        bpy.ops.mesh.select_all(action='SELECT')
        bpy.ops.mesh.remove_doubles()
        bpy.ops.mesh.dissolve_limited(angle_limit=0.001)
        bpy.ops.object.mode_set(mode='OBJECT')

        return {"FINISHED"}


class LSystemPanel(bpy.types.Panel):
    # Where to add the panel in the UI
    bl_space_type = "VIEW_3D"  # 3D Viewport area
    bl_region_type = "UI"  # Sidebar region

    bl_idname = "VIEW3D_PT_create_l_system"  # Callable Id for panel
    bl_category = "L-System"  # Found in the sidebar
    bl_label = "L-System"  # Found at the top of the panel

    bpy.types.Scene.show_subpanel = bpy.props.BoolProperty(name="show_subpanel")

    def draw(self, context):
        """Define the layout of the panel"""
        scene = context.scene
        lsystem_props = scene.lsystem_props
        show_subpanel = scene.show_subpanel

        self.layout.prop(lsystem_props, "preset_options", text="Select Preset", icon="PRESET")
        self.layout.separator()

        column = self.layout.column()
        column.operator("mesh.apply_presets", text="Use Selected Preset", icon="CHECKMARK")
        self.layout.separator(type='LINE')
        self.layout.separator(factor=0.1)

        row_4 = self.layout.row()
        row_4.prop(lsystem_props, "shape_toggle", icon="MESH_CYLINDER")
        if lsystem_props.use_tube:
            self.layout.prop(lsystem_props, "tube_resolution", icon="SEQ_CHROMA_SCOPE")
            self.layout.prop(lsystem_props, "tube_radius", icon="CON_SIZELIKE")
        self.layout.separator()

        row_3 = self.layout.row()
        row_3.label(text="Enter rules for custom L-System", icon="CURRENT_FILE")
        self.layout.prop(lsystem_props, "premise", icon="RIGHTARROW_THIN")
        self.layout.separator()

        self.layout.prop(lsystem_props, "rules_1", icon="RIGHTARROW_THIN")
        self.layout.prop(lsystem_props, "rules_2", icon="RIGHTARROW_THIN")
        self.layout.prop(lsystem_props, "rules_3", icon="RIGHTARROW_THIN")
        self.layout.prop(lsystem_props, "rules_4", icon="RIGHTARROW_THIN")
        self.layout.separator()

        self.layout.prop(lsystem_props, "l_sys_angle")
        self.layout.prop(lsystem_props, "iterations")
        self.layout.prop(lsystem_props, "step_length")

        row_1 = self.layout.row()
        row_1.label(text="Using too many iterations may cause Blender to crash", icon="ERROR")

        subpanel = self.layout.row()
        subpanel.prop(scene, "show_subpanel", text="Custom Variables",
                      icon="DOWNARROW_HLT" if show_subpanel else "RIGHTARROW", emboss=False)

        if show_subpanel:
            box = self.layout.box()
            box.prop(lsystem_props, "custom_b")
            box.prop(lsystem_props, "custom_c")
            box.prop(lsystem_props, "custom_d")
            box.prop(lsystem_props, "custom_q")
            box.prop(lsystem_props, "custom_p")
            box.prop(lsystem_props, "custom_r")

        self.layout.separator(factor=0.1)
        self.layout.separator(type='LINE')
        self.layout.separator(factor=0.1)

        row_2 = self.layout.row()
        row_2.operator("mesh.create_l_sys", text="Create L-System", icon="SYSTEM")

        creator = self.layout.row()
        creator.label(text="Made by Rut Bivrin", icon="GHOST_ENABLED")


# _______________________________________________________________________________________________________________________________________
# Register and unregister all classes
# _______________________________________________________________________________________________________________________________________

def register():
    bpy.utils.register_class(LSystemProperties)
    bpy.types.Scene.lsystem_props = bpy.props.PointerProperty(type=LSystemProperties)
    bpy.types.Scene.show_subpanel = bpy.props.BoolProperty(name="show_subpanel")
    bpy.utils.register_class(LSystemPanel)
    bpy.utils.register_class(ApplyPresets)
    bpy.utils.register_class(CreateLSystem)


def unregister():
    bpy.utils.unregister_class(CreateLSystem)
    bpy.utils.unregister_class(ApplyPresets)
    bpy.utils.unregister_class(LSystemPanel)
    del bpy.types.Scene.show_subpanel
    del bpy.types.Scene.lsystem_props
    bpy.utils.unregister_class(LSystemProperties)


if __name__ == "__main__":
    register()