import typing

import maya.api.OpenMaya as om
import maya.cmds as mc

import templates as templates


def add_angle_attribute(object_: om.MObject, name: str, min_: float = None, max_: float = None, readable: bool = True,
                        writable: bool = True, storable: bool = True, keyable: bool = True) -> None:
    """
    creates an attribute of type double angle
    """
    if min_ is not None and max_ is not None:
        mc.addAttr(get_name(object_), longName=name, attributeType="doubleAngle", minValue=min_, maxValue=max_,
                   readable=readable, writable=writable, storable=storable, keyable=keyable)
        return
    if min_ is not None:
        mc.addAttr(get_name(object_), longName=name, attributeType="doubleAngle", minValue=min_, readable=readable,
                   writable=writable, storable=storable, keyable=keyable)
        return
    if max_ is not None:
        mc.addAttr(get_name(object_), longName=name, attributeType="doubleAngle", maxValue=max_, readable=readable,
                   writable=writable, storable=storable, keyable=keyable)
        return

    mc.addAttr(get_name(object_), longName=name, attributeType="doubleAngle", readable=readable, writable=writable,
               storable=storable, keyable=keyable)


def add_double_attribute(object_: om.MObject, name: str, min_: float = None, max_: float = None, readable: bool = True,
                         writable: bool = True, storable: bool = True, keyable: bool = True) -> None:
    """
    creates an attribute of type double
    """
    if min_ is not None and max_ is not None:
        mc.addAttr(get_name(object_), longName=name, attributeType="double", minValue=min_, maxValue=max_,
                   readable=readable, writable=writable, storable=storable, keyable=keyable)
        return
    if min_ is not None:
        mc.addAttr(get_name(object_), longName=name, attributeType="double", minValue=min_, readable=readable,
                   writable=writable, storable=storable, keyable=keyable)
        return
    if max_ is not None:
        mc.addAttr(get_name(object_), longName=name, attributeType="double", maxValue=max_, readable=readable,
                   writable=writable, storable=storable, keyable=keyable)
        return

    mc.addAttr(get_name(object_), longName=name, attributeType="double", readable=readable, writable=writable,
               storable=storable, keyable=keyable)


def add_enum_attribute(object_: om.MObject, name: str, list_: list, readable: bool = True, writable: bool = True,
                       storable: bool = True, keyable: bool = True) -> None:
    """
    creates an attribute of type enum
    """
    enum_names = ""
    for element in list_:
        enum_names += element + ":"

    enum_names = enum_names[:-1]

    mc.addAttr(get_name(object_), longName=name, attributeType="enum", enumName=enum_names, readable=readable,
               writable=writable, storable=storable, keyable=keyable)


def add_to_set(set_: str, object_: om.MObject) -> None:
    """
    adds the given object ot the set
    """
    if not (mc.objExists(set_) and mc.objectType(set_, isType="objectSet")):
        if mc.objExists(set_):
            mc.delete(set_)

        mc.sets(name=set_, empty=True)
    mc.sets(get_name(object_), addElement=set_)


def bind_surface(surface: om.MObject, joints: typing.List[om.MObject]) -> om.MObject:
    """
    binds a surface to joints
    """
    objects = [get_name(surface)]
    objects.extend([get_name(j) for j in joints])

    cluster = get_object(mc.skinCluster(objects, toSelectedBones=True, skinMethod=0, maximumInfluences=2,
                                        normalizeWeights=2, dropoffRate=5.0)[0])

    return cluster


def calculate_aim_vector(position1: om.MVector, position2: om.MVector, normalized: bool = True) -> om.MVector:
    """
    calculates the aim vector from one position to another
    """
    aim = (position2 - position1)

    if normalized:
        aim = aim.normalize()

    return aim


def calculate_distance_between(position1: om.MVector, position2: om.MVector) -> float:
    """
    calculates the distance between two positions
    """
    distance = (position2 - position1).length()

    return distance


def calculate_even_spacing(point1: om.MVector, point2: om.MVector, segments: int) -> float:
    distance = calculate_distance_between(point1, point2)
    spacing = distance / float(segments)

    return spacing


def calculate_mid_point(position1: om.MVector, position2: om.MVector) -> om.MVector:
    """
    calculates the mid-point between two positions
    """
    mid_point = (position1 + position2) / 2

    return mid_point


def calculate_orientation_from_axis(x_axis: om.MVector, y_axis: om.MVector, z_axis: om.MVector) -> om.MQuaternion:
    """
    calculates the rotation from three axis
    """
    rotation = om.MTransformationMatrix(create_rotation_matrix_from_axis(x_axis, y_axis, z_axis)).rotation(
        asQuaternion=True)

    return rotation


def calculate_orientation_from_positions(position1: om.MVector, position2: om.MVector,
                                         position3: om.MVector) -> om.MQuaternion:
    """
    calculates the rotation from three positions
    """
    x_axis = (position2 - position1).normalize()
    z_axis = ((position2 - position1).normalize() ^ (position3 - position1).normalize()).normalize()
    y_axis = (x_axis ^ z_axis).normalize()

    rotation = calculate_orientation_from_axis(x_axis, y_axis, z_axis)

    return rotation


def calculate_three_chain_pole_vector(start_position: om.MVector, pole_position: om.MVector,
                                      parent_rotation: om.MQuaternion) -> om.MVector:
    """
    calculates the pole vector for a three chain
    """
    dir_ = pole_position - start_position
    d_matrix = compose_matrix(translation=dir_)
    pr_matrix = compose_matrix(rotation=parent_rotation).inverse()

    pv_matrix = om.MTransformationMatrix(d_matrix * pr_matrix)
    pv_translation = pv_matrix.translation(om.MSpace.kWorld)

    return pv_translation


def calculate_two_chain_pole_vector(world_start: om.MMatrix, world_parent: om.MMatrix) -> om.MVector:
    """
    calculates the pole vector for a two chain
    """
    pv_matrix = compose_matrix(translation=om.MVector.kYnegAxisVector) * world_start * world_parent.inverse()
    pv_translation = get_translation(pv_matrix)

    return pv_translation


def calculate_xz_orientation(position1: om.MVector, position2: om.MVector, position3: om.MVector) -> om.MQuaternion:
    """
    calculates the x and z rotation from three position and has y pointing upwards
    """
    y_axis = om.MVector.kYaxisVector
    z_axis = ((position2 - position1).normalize() ^ (position3 - position1).normalize()).normalize()
    x_axis = (y_axis ^ z_axis).normalize()

    rotation = om.MTransformationMatrix(create_rotation_matrix_from_axis(x_axis, y_axis, z_axis)).rotation(
        asQuaternion=True)

    return rotation


def compose_matrix(translation: om.MVector = om.MVector.kZeroVector,
                   rotation: om.MQuaternion = om.MQuaternion.kIdentity, scale: tuple = (1, 1, 1)) -> om.MMatrix:
    """
    composes a matrix from individual components
    """
    t_matrix = om.MTransformationMatrix()

    t_matrix.setTranslation(translation, om.MSpace.kWorld)
    t_matrix.setRotation(rotation)
    t_matrix.setScale(scale, om.MSpace.kWorld)

    return t_matrix.asMatrix()


def connect_plugs(out_object: om.MObject, out_attribute: str, in_object: om.MObject, in_attribute: str,
                  out_i: int = None, out_c_attribute: str = None, in_i: int = None, in_c_attribute: str = None) -> None:
    """
    connects two plugs
    """
    dg_mod = om.MDGModifier()

    dg_mod.connect(get_plug(out_object, out_attribute, out_i, out_c_attribute),
                   get_plug(in_object, in_attribute, in_i, in_c_attribute))
    dg_mod.doIt()


def convert_to_local_space(world: om.MMatrix, parent_: om.MMatrix) -> om.MMatrix:
    """
    converts the world space matrix to local space
    """
    local = world * parent_.inverse()

    return local


def create_circle_control(name: str, radius: float, normal: typing.Tuple[int, int, int], parent_: om.MObject = None,
                          set_: str = None) -> om.MObject:
    """
    creates a circle control
    """
    control = get_object(mc.circle(normal=normal, center=(0, 0, 0), radius=radius)[0])
    rename(control, name)

    if parent_ is not None:
        parent(control, parent_)

    if set_ is not None:
        add_to_set(set_, control)

    return control


def create_compound_3numeric_attribute(node: typing.Type[templates.BaseNode], name: str, type_: om.MFnNumericData,
                                       is_input: bool, is_output: bool, is_array: bool,
                                       suffixes: typing.List[str] = ("X", "Y", "Z")) -> typing.Tuple[
        om.MObject, om.MObject, om.MObject, om.MObject]:
    """
    creates an attribute of type vector
    """
    fn_c_attr = om.MFnCompoundAttribute()
    fn_attr = om.MFnNumericAttribute()

    obj = fn_c_attr.create(name, name)
    fn_c_attr.writable = is_input
    fn_c_attr.readable = is_output
    fn_c_attr.array = is_array
    fn_c_attr.usesArrayDataBuilder = is_array

    child1 = fn_attr.create(name + suffixes[0], name + suffixes[0], type_)
    fn_attr.writable = is_input
    fn_attr.readable = is_output
    fn_c_attr.addChild(child1)
    child2 = fn_attr.create(name + suffixes[1], name + suffixes[1], type_)
    fn_attr.writable = is_input
    fn_attr.readable = is_output
    fn_c_attr.addChild(child2)
    child3 = fn_attr.create(name + suffixes[2], name + suffixes[2], type_)
    fn_attr.writable = is_input
    fn_attr.readable = is_output
    fn_c_attr.addChild(child3)

    node.addAttribute(obj)

    return obj, child1, child2, child3


def create_compound_3unit_attribute(node: typing.Type[templates.BaseNode], name: str, type_: om.MFnUnitAttribute,
                                    is_input: bool, is_output: bool, is_array: bool,
                                    suffixes: typing.List[str] = ("X", "Y", "Z")) -> typing.Tuple[
        om.MObject, om.MObject, om.MObject, om.MObject]:
    """
    creates an attribute of type vector
    """
    fn_c_attr = om.MFnCompoundAttribute()
    fn_attr = om.MFnUnitAttribute()

    obj = fn_c_attr.create(name, name)
    fn_c_attr.writable = is_input
    fn_c_attr.readable = is_output
    fn_c_attr.array = is_array
    fn_c_attr.usesArrayDataBuilder = is_array

    child1 = fn_attr.create(name + suffixes[0], name + suffixes[0], type_)
    fn_attr.writable = is_input
    fn_attr.readable = is_output
    fn_c_attr.addChild(child1)
    child2 = fn_attr.create(name + suffixes[1], name + suffixes[1], type_)
    fn_attr.writable = is_input
    fn_attr.readable = is_output
    fn_c_attr.addChild(child2)
    child3 = fn_attr.create(name + suffixes[2], name + suffixes[2], type_)
    fn_attr.writable = is_input
    fn_attr.readable = is_output
    fn_c_attr.addChild(child3)

    node.addAttribute(obj)

    return obj, child1, child2, child3


def create_cubic_curve(points: typing.List[om.MVector]) -> om.MObject:
    """
    creates a curve
    """
    point = [(v.x, v.y, v.z) for v in points]
    knot = []

    ns = len(points) - 3
    nk = len(points) + 3 - 1

    if ns <= 0:
        knot.extend([0 for _ in range(nk)])
    else:
        knot.extend([0, 0, 0])
        knot.extend([i for i in range(1, nk - 6 + 1)])
        knot.extend([ns, ns, ns])

    curve = get_object(mc.curve(degree=3, point=point, knot=knot))

    return curve


def create_diamond_control(name: str, radius: float, normal: typing.Tuple[int, int, int], parent_: om.MObject = None,
                           set_: str = None) -> om.MObject:
    """
    creates a circle control
    """
    control = get_object(mc.circle(normal=normal, center=(0, 0, 0), sections=4, degree=1, radius=radius)[0])
    rename(control, name)

    if parent_ is not None:
        parent(control, parent_)

    if set_ is not None:
        add_to_set(set_, control)

    return control


def create_group(name: str, parent_: om.MObject = None, set_: str = None) -> om.MObject:
    """
    creates an empty group
    """
    group = get_object(mc.group(name=name, empty=True))

    if parent_ is not None:
        parent(group, parent_)

    if set_ is not None:
        add_to_set(set_, group)

    return group


def create_heptagon_control(name: str, radius: float, normal: typing.Tuple[int, int, int], parent_: om.MObject = None,
                            set_: str = None) -> om.MObject:
    """
    creates a circle control
    """
    control = get_object(mc.circle(normal=normal, center=(0, 0, 0), sections=7, degree=1, radius=radius)[0])
    rename(control, name)

    if parent_ is not None:
        parent(control, parent_)

    if set_ is not None:
        add_to_set(set_, control)

    return control


def create_ik_handle(name: str, start: om.MObject, end: om.MObject, parent_: om.MObject = None,
                     set_: str = None) -> om.MObject:
    """
    creates an ik handle
    """
    handle = get_object(mc.ikHandle(name=name, startJoint=get_name(start), endEffector=get_name(end))[0])
    reset_default_plugs(handle)

    if parent_ is not None:
        parent(handle, parent_)

    if set_ is not None:
        add_to_set(set_, handle)

    return handle


def create_joint(name: str, parent_: om.MObject = None, set_: str = None) -> om.MObject:
    """
    creates a joint
    """
    mc.select(clear=True)

    joint = get_object(mc.joint(name=name))

    mc.select(clear=True)

    if parent_ is not None:
        parent(joint, parent_)

    if set_ is not None:
        add_to_set(set_, joint)

    return joint


def create_linear_curve(start: om.MVector, end: om.MVector) -> om.MObject:
    """
    creates a curve
    """
    point = [(start.x, start.y, start.z), (end.x, end.y, end.z)]
    knot = [0, 1]

    curve = get_object(mc.curve(degree=1, point=point, knot=knot))

    return curve


def create_locator(name: str, parent_: om.MObject = None, set_: str = None) -> om.MObject:
    """
    creates a locator
    """
    locator = get_object(mc.spaceLocator(name=name)[0])

    if parent_ is not None:
        parent(locator, parent_)

    if set_ is not None:
        add_to_set(set_, locator)

    return locator


def create_matrix_attribute(node: typing.Type[templates.BaseNode], name: str, is_input: bool, is_output: bool,
                            is_array: bool) -> om.MObject:
    """
    creates an attribute of type matrix
    """
    fn_attr = om.MFnMatrixAttribute()

    obj = fn_attr.create(name, name)
    fn_attr.writable = is_input
    fn_attr.readable = is_output
    fn_attr.array = is_array
    fn_attr.usesArrayDataBuilder = is_array

    node.addAttribute(obj)

    return obj


def create_message_attribute(node: typing.Type[templates.BaseNode], name: str, is_input: bool, is_output: bool,
                             is_array: bool) -> om.MObject:
    """
    creates an attribute of type message
    """
    fn_attr = om.MFnMessageAttribute()

    obj = fn_attr.create(name, name)
    fn_attr.writable = is_input
    fn_attr.readable = is_output
    fn_attr.array = is_array
    fn_attr.usesArrayDataBuilder = is_array

    node.addAttribute(obj)

    return obj


def create_node(type_: str, name: str, set_: str = None) -> om.MObject:
    """
    creates a node
    """
    node = get_object(mc.createNode(type_, name=name))

    if set_ is not None:
        add_to_set(set_, node)

    return node


def create_numeric_attribute(node: typing.Type[templates.BaseNode], name: str, type_: om.MFnNumericData, is_input: bool,
                             is_output: bool, is_array: bool) -> om.MObject:
    """
    creates an attribute of type double
    """
    fn_attr = om.MFnNumericAttribute()

    obj = fn_attr.create(name, name, type_)
    fn_attr.writable = is_input
    fn_attr.readable = is_output
    fn_attr.array = is_array
    fn_attr.usesArrayDataBuilder = is_array

    node.addAttribute(obj)

    return obj


def create_rotation_matrix_from_axis(x_axis: om.MVector, y_axis: om.MVector, z_axis: om.MVector) -> om.MMatrix:
    """
    creates a rotation matrix given three axis
    """
    matrix = om.MMatrix([x_axis[0], x_axis[1], x_axis[2], 0,
                         y_axis[0], y_axis[1], y_axis[2], 0,
                         z_axis[0], z_axis[1], z_axis[2], 0,
                         0, 0, 0, 1])

    return matrix


def create_surface(curve1: om.MObject, curve2: om.MObject) -> om.MObject:
    """
    creates a surface
    """
    surface = get_object(
        mc.loft(get_name(curve1), get_name(curve2), constructionHistory=True, uniform=True, close=False,
                autoReverse=True, degree=3, sectionSpans=1, range=False, polygon=0)[0])

    return surface


def create_triangle_control(name: str, radius: float, normal: typing.Tuple[int, int, int], parent_: om.MObject = None,
                            set_: str = None) -> om.MObject:
    """
    creates a circle control
    """
    control = get_object(mc.circle(normal=normal, center=(0, 0, 0), sections=3, degree=1, radius=radius)[0])
    rename(control, name)

    if parent_ is not None:
        parent(control, parent_)

    if set_ is not None:
        add_to_set(set_, control)

    return control


def create_typed_attribute(node: typing.Type[templates.BaseNode], name: str, type_: om.MFnData, is_input: bool,
                           is_output: bool, is_array: bool) -> om.MObject:
    """
    creates an attribute of the given type
    """
    fn_attr = om.MFnTypedAttribute()

    obj = fn_attr.create(name, name, type_)
    fn_attr.writable = is_input
    fn_attr.readable = is_output
    fn_attr.array = is_array
    fn_attr.usesArrayDataBuilder = is_array

    node.addAttribute(obj)

    return obj


def create_unit_attribute(node: typing.Type[templates.BaseNode], name: str, type_: om.MFnUnitAttribute, is_input: bool,
                          is_output: bool, is_array: bool) -> om.MObject:
    """
    creates an attribute of type double
    """
    fn_attr = om.MFnUnitAttribute()

    obj = fn_attr.create(name, name, type_)
    fn_attr.writable = is_input
    fn_attr.readable = is_output
    fn_attr.array = is_array
    fn_attr.usesArrayDataBuilder = is_array

    node.addAttribute(obj)

    return obj


def get_angle_attribute(block: om.MDataBlock, object_: om.MObject) -> om.MAngle:
    """
    gets the value of an attribute of type angle
    """
    angle = block.inputValue(object_).asAngle()

    return angle


def get_double_attribute(block: om.MDataBlock, object_: om.MObject) -> float:
    """
    gets the value of an attribute of type double
    """
    value = block.inputValue(object_).asDouble()

    return value


def get_int_attribute(block: om.MDataBlock, object_: om.MObject) -> int:
    """
    gets the value of an attribute of type int
    """
    value = block.inputValue(object_).asInt()

    return value


def get_matrix_attribute(block: om.MDataBlock, object_: om.MObject) -> typing.Tuple[om.MMatrix, om.MVector,
                                                                                    om.MQuaternion, typing.Tuple[
                                                                                        float, float, float]]:
    """
    gets the value of an attribute of type matrix
    """
    matrix = block.inputValue(object_).asMatrix()

    t_matrix = om.MTransformationMatrix(matrix)

    translation = t_matrix.translation(om.MSpace.kWorld)
    rotation = t_matrix.rotation(asQuaternion=True)
    scale = t_matrix.scale(om.MSpace.kWorld)

    return matrix, translation, rotation, scale


def get_matrix_attribute_in_array(block: om.MDataBlock, object_: om.MObject,
                                  index: int) -> typing.Tuple[om.MMatrix, om.MVector, om.MQuaternion,
                                                              typing.Tuple[float, float, float]]:
    """
    gets the value of an element of type matrix from an array attribute
    """
    array = block.inputArrayValue(object_)

    try:
        array.jumpToPhysicalElement(index)
    except:
        matrix = om.MMatrix.kIdentity
        translation = om.MVector.kZeroVector
        rotation = om.MQuaternion.kIdentity
        scale = (1, 1, 1)
    else:
        matrix = array.inputValue().asMatrix()

        t_matrix = om.MTransformationMatrix(matrix)

        translation = t_matrix.translation(om.MSpace.kWorld)
        rotation = t_matrix.rotation(asQuaternion=True)
        scale = t_matrix.scale(om.MSpace.kWorld)

    return matrix, translation, rotation, scale


def get_name(object_: om.MObject) -> str:
    """
    gets the name of the given object
    """
    name = om.MFnDependencyNode(object_).name()

    return name


def get_object(name: str) -> om.MObject:
    """
    gets the object with the given name
    """
    sel = om.MSelectionList()
    sel.add(name)

    obj = sel.getDependNode(0)

    return obj


def get_plug(object_: om.MObject, attribute: str, i: int = None, c_attribute: str = None) -> om.MPlug:
    """
    gets a plug
    """
    dep_obj = om.MFnDependencyNode(object_)

    if i is None:
        plug = dep_obj.findPlug(attribute, False)
    else:
        plug = dep_obj.findPlug(attribute, False).elementByLogicalIndex(i)

    if c_attribute is None or not plug.isCompound:
        return plug
    else:
        return plug.child(c_attribute)


def get_rotation(matrix: om.MMatrix) -> om.MQuaternion:
    """
    gets the rotation of the given matrix
    """
    rotation = om.MTransformationMatrix(matrix).rotation(asQuaternion=True)

    return rotation


def get_rotation_plugs(obj: om.MObject) -> typing.List[om.MPlug]:
    """
    gets all rotation plugs
    """
    if not obj.hasFn(om.MFn.kDependencyNode):
        return []

    fn = om.MFnDependencyNode(obj)

    names = ('rx', 'ry', 'rz')
    plugs = [fn.findPlug(_name, False) for _name in names]

    return plugs


def get_scale(matrix: om.MMatrix) -> typing.Tuple[float, float, float]:
    """
    gets the scale of the given matrix
    """
    scale = om.MTransformationMatrix(matrix).scale(om.MSpace.kWorld)

    return scale


def get_scale_plugs(obj: om.MObject) -> typing.List[om.MPlug]:
    """
    gets all scale plugs
    """
    if not obj.hasFn(om.MFn.kDependencyNode):
        return []

    fn = om.MFnDependencyNode(obj)

    names = ('sx', 'sy', 'sz')
    plugs = [fn.findPlug(_name, False) for _name in names]

    return plugs


def get_selection() -> typing.List[om.MObject]:
    """
    gets the viewport selection
    """
    names = mc.ls(selection=True)

    objects = [get_object(name) for name in names]

    return objects


def get_shape(object_: om.MObject) -> om.MObject:
    """
    gets the shape node
    """
    sp = om.MFnDagNode(object_).getPath().extendToShape()
    shape = sp.node()

    return shape


def get_surface_attribute(block: om.MDataBlock, object_: om.MObject) -> int:
    """
    gets the value of an attribute of type surface
    """
    value = block.inputValue(object_).asNurbsSurface()

    return value


def get_translation(matrix: om.MMatrix) -> om.MVector:
    """
    gets the translation of the given matrix
    """
    translation = om.MTransformationMatrix(matrix).translation(om.MSpace.kWorld)

    return translation


def get_translation_plugs(obj: om.MObject) -> typing.List[om.MPlug]:
    """
    gets all translation plugs
    """
    if not obj.hasFn(om.MFn.kDependencyNode):
        return []

    fn = om.MFnDependencyNode(obj)

    names = ('tx', 'ty', 'tz')
    plugs = [fn.findPlug(_name, False) for _name in names]

    return plugs


def lock_attributes(object_: om.MObject, attributes: typing.List[str]) -> None:
    """
    locks attributes
    """
    for attribute in attributes:
        name = get_name(object_) + "." + attribute
        mc.setAttr(name, lock=True)


def parent(object_: om.MObject, parent_: om.MObject) -> None:
    """
    parents a given node under a given parent
    """
    dag_mod = om.MDagModifier()

    dag_mod.reparentNode(object_, parent_)
    dag_mod.doIt()


def pick_matrix(matrix: om.MMatrix, pick_translation: bool = False, pick_rotation: bool = False,
                pick_scale: bool = False) -> om.MMatrix:
    """
    picks the elements from the matrix
    """
    translation = get_translation(matrix) if pick_translation else om.MVector.kZeroVector
    rotation = get_rotation(matrix) if pick_rotation else om.MQuaternion.kIdentity
    scale = get_scale(matrix) if pick_scale else (1, 1, 1)

    pm = compose_matrix(translation=translation, rotation=rotation, scale=scale)

    return pm


def rebuild_surface(surface: om.MObject) -> om.MObject:
    """
    rebuilds a surface
    """
    surface_ = get_object(mc.rebuildSurface(get_name(surface), constructionHistory=False, replaceOriginal=True,
                                            rebuildType=0, keepRange=0, keepControlPoints=False, keepCorners=False,
                                            spansU=1, degreeU=3, spansV=4, degreeV=3)[0])

    return surface_


def remove_callbacks(obj: om.MObject) -> None:
    """
    removes all callbacks from the given object.
    """
    cbs = om.MMessage.nodeCallbacks(obj)
    for cb in cbs:
        om.MMessage.removeCallback(cb)


def rename(object_: om.MObject, name: str) -> None:
    """
    renames a given node
    """
    dg_mod = om.MDGModifier()

    dg_mod.renameNode(object_, name)
    dg_mod.doIt()


def reset_default_plugs(object_: om.MObject):
    """
    resets the default plugs
    """
    set_double_plug(object_, "translateX", 0.0)
    set_double_plug(object_, "translateY", 0.0)
    set_double_plug(object_, "translateZ", 0.0)

    set_double_plug(object_, "rotateX", 0.0)
    set_double_plug(object_, "rotateY", 0.0)
    set_double_plug(object_, "rotateZ", 0.0)

    set_double_plug(object_, "scaleX", 1.0)
    set_double_plug(object_, "scaleY", 1.0)
    set_double_plug(object_, "scaleZ", 1.0)


def rotate_vector(vector: om.MVector, quaternion: om.MQuaternion) -> om.MVector:
    """
    rotates a vector by a quaternion
    """
    rotated = get_translation(compose_matrix(translation=vector) * compose_matrix(rotation=quaternion))

    return rotated


def set_3double_attribute_from_quaternion(block: om.MDataBlock, object_: om.MObject, value: om.MQuaternion) -> None:
    """
    sets the value of an attribute of type numeric 3 double
    """
    euler = value.asEulerRotation()
    handle = block.outputValue(object_)
    handle.set3Double(euler.x, euler.y, euler.z)
    handle.setClean()


def set_3double_attribute_from_sequence(block: om.MDataBlock, object_: om.MObject, value: typing.Tuple[float,
                                                                                                       float,
                                                                                                       float]) -> None:
    """
    sets the value of an attribute of type numeric 3 double
    """
    handle = block.outputValue(object_)
    handle.set3Double(value[0], value[1], value[2])
    handle.setClean()


def set_3double_attribute_from_vector(block: om.MDataBlock, object_: om.MObject, value: om.MVector) -> None:
    """
    sets the value of an attribute of type numeric 3 double
    """
    handle = block.outputValue(object_)
    handle.set3Double(value.x, value.y, value.z)
    handle.setClean()


def set_attribute_affects(node: typing.Type[templates.BaseNode], *args: om.MObject) -> None:
    """
    sets the attribute affects
    """
    affected = args[0]
    affects = args[1:]

    for affect in affects:
        node.attributeAffects(affect, affected)


def set_compound_3angle_attribute_from_quaternion(block: om.MDataBlock, object_: typing.Tuple[om.MObject, om.MObject,
                                                                                              om.MObject, om.MObject],
                                                  value: om.MQuaternion) -> None:
    """
    sets the value of an attribute of type compound angle
    """
    euler = value.asEulerRotation()
    handle = block.outputValue(object_[0])

    handle.child(object_[1]).setMAngle(om.MAngle(euler.x, om.MAngle.kRadians))
    handle.child(object_[1]).setClean()
    handle.child(object_[2]).setMAngle(om.MAngle(euler.y, om.MAngle.kRadians))
    handle.child(object_[2]).setClean()
    handle.child(object_[3]).setMAngle(om.MAngle(euler.z, om.MAngle.kRadians))
    handle.child(object_[3]).setClean()

    handle.setClean()


def set_compound_3distance_attribute_from_vector(block: om.MDataBlock, object_: typing.Tuple[om.MObject, om.MObject,
                                                                                             om.MObject, om.MObject],
                                                 value: om.MVector) -> None:
    """
    sets the value of an attribute of type compound distance
    """
    handle = block.outputValue(object_[0])

    handle.child(object_[1]).setMDistance(om.MDistance(value.x))
    handle.child(object_[1]).setClean()
    handle.child(object_[2]).setMDistance(om.MDistance(value.y))
    handle.child(object_[2]).setClean()
    handle.child(object_[3]).setMDistance(om.MDistance(value.z))
    handle.child(object_[3]).setClean()

    handle.setClean()


def set_compound_3double_attribute_from_sequence(block: om.MDataBlock, object_: typing.Tuple[om.MObject, om.MObject,
                                                                                             om.MObject, om.MObject],
                                                 value: typing.Tuple[float, float, float]) -> None:
    """
    sets the value of an attribute of type compound double
    """
    handle = block.outputValue(object_[0])

    handle.child(object_[1]).setDouble(value[0])
    handle.child(object_[1]).setClean()
    handle.child(object_[2]).setDouble(value[1])
    handle.child(object_[2]).setClean()
    handle.child(object_[3]).setDouble(value[2])
    handle.child(object_[3]).setClean()

    handle.setClean()


def set_compound_3double_attribute_from_vector(block: om.MDataBlock, object_: typing.Tuple[om.MObject, om.MObject,
                                                                                           om.MObject, om.MObject],
                                               value: om.MVector) -> None:
    """
    sets the value of an attribute of type compound double
    """
    handle = block.outputValue(object_[0])

    handle.child(object_[1]).setDouble(value.x)
    handle.child(object_[1]).setClean()
    handle.child(object_[2]).setDouble(value.y)
    handle.child(object_[2]).setClean()
    handle.child(object_[3]).setDouble(value.z)
    handle.child(object_[3]).setClean()

    handle.setClean()


def set_double_plug(object_: om.MObject, attribute: str, value: float, i: int = None, c_attribute: str = None) -> None:
    """
    sets a plugs value as a double
    """
    get_plug(object_, attribute, i, c_attribute).setDouble(value)


def set_matrix_attribute(block: om.MDataBlock, object_: om.MObject, value: om.MMatrix) -> None:
    """
    sets the value of an attribute of type matrix
    """
    handle = block.outputValue(object_)
    handle.setMMatrix(value)
    handle.setClean()


def transform_in_space(transform: om.MMatrix, space: om.MMatrix, default: om.MMatrix = None) -> om.MMatrix:
    """
    creates a matrix that translates
    """
    mt_pd = space.inverse() * transform * space

    mt = mt_pd if default is None else default * mt_pd

    return mt


def translate_in_space(translation: om.MVector, space: om.MQuaternion, default: om.MVector = None) -> om.MVector:
    """
    creates a matrix that translates
    """
    tr_mt = compose_matrix(translation=translation)
    sp_mt = compose_matrix(rotation=space)
    mt = sp_mt.inverse() * tr_mt * sp_mt

    tr = default + get_translation(mt) if default is not None else get_translation(mt)

    return tr
