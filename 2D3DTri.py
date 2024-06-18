import math
import numpy as np

""" A robust intersection checker of two triangles whether in 2D or 3D - call primary function at the end of the file for different uses """
## FOR 2D
def find_orientation(p,q,r):
    """
    pq is the line-segment and we want to determine on which side of it r is located
    0 means they are colinear, 1 right turn, -1 left turn
    """#
    vec_pr = r-p
    vec_pq = q-p
    my_matrix = np.vstack((vec_pr,vec_pq))
    my_det = np.linalg.det(my_matrix) # sign of det of a matrix with the vector as its rows
    if np.abs(my_det) < 1e-10:
        return 0
    return np.sign(my_det)

def on_line_segment(p,q,r):
    """
    assuming the points are co-linear, check if also r is on the line-segment pq
    """#
    return (min(p[0], q[0]) <= r[0] <= max(p[0], q[0])) and (min(p[1], q[1]) <= r[1] <= max(p[1], q[1]))

def segments_intersect(s1,e1,s2,e2):
    """
    self-explanatory. utilizing orientation and taking care of special cases 
    """#
    first_orientation = find_orientation(s1,e1,s2)
    second_orientation = find_orientation(s1,e1,e2)
    third_orientation = find_orientation(s2,e2,s1)
    fourth_orientation = find_orientation(s2,e2,e1)

    if (first_orientation==0 and on_line_segment(s1,e1,s2)): ## !! I'm pretty sure the order here in conj with orientation is good
        return True

    if (second_orientation==0 and on_line_segment(s1,e1,e2)):
        return True

    if (third_orientation==0 and on_line_segment(s2,e2,s1)):
        return True

    if (fourth_orientation==0 and on_line_segment(s2,e2,e1)):
        return True

    if ((first_orientation != second_orientation) and (third_orientation != fourth_orientation)):
        return True
    return False

def is_inscribed(segments,vertices):
    """
    Two triangles can interesect without their edges intersecting. Checking here if one triangle is strictly
    inscribed in another triangle. Vertics belong to the suspect inscribed triangle, 
    and segments of the suspect inscribing triangle
    """#
    for i in range(len(vertices)):
        vertex = vertices[i]
        check_list = []
        for j in range(len(segments)):
            segment = segments[j]
            check_list.append(find_orientation(segment[0],segment[1],vertex))
        if (not (check_list[0]==check_list[1] and check_list[1]==check_list[2])):
            return False
    return True

def two_2D_triangles_intersect(vertices1, vertices2):
    """
    Our primary function for checking intersection of two 2D triangles.
    """#
    vertices1 = np.array(vertices1)
    vertices2 = np.array(vertices2)

    ## whether segments intersect
    segments1 = np.array([[vertices1[0], vertices1[1]], [vertices1[1], vertices1[2]], [vertices1[2], vertices1[0]]])
    segments2 = np.array([[vertices2[0], vertices2[1]], [vertices2[1], vertices2[2]], [vertices2[2], vertices2[0]]])
    for i in range(len(segments1)):
        for j in range(len(segments2)):
            if (segments_intersect(segments1[i][0],segments1[i][1],segments2[j][0],segments2[j][1])):
                return True

    # assuming no segment intersect with any other segment
    # whether one triangle is strictly inscribed inside the other
    return is_inscribed(segments1,vertices2) or is_inscribed(segments2,vertices1)

## FOR 2D

## FOR 3D
def rotate_to_same_z(vertices,point):
    """
    when we have a triangle, and a point (which is a segment intersection with the plane of the triangle) that are
    coplanar and we want to rotate them to a plane parallel to xy-plane so we can reduce their dimension to 2D and
    use the good old 2D functions from above
    """#
    v1,v2,v3 = vertices
    vec12 = v2-v1
    vec13 = v3-v1
    normal = np.cross(vec12,vec13) # our normal vector
    normal = np.divide(normal, np.linalg.norm(normal))

    all_points = np.array([v1,v2,v3,point])
    theta_x = np.arctan2(normal[1], normal[2])
    theta_y = np.arctan2(-normal[0], np.sqrt(normal[1] ** 2 + normal[2] ** 2))

    # Creating rotation matrices for each axis
    rotation_matrix_x = np.array([[1, 0, 0],
                                  [0, np.cos(theta_x), -np.sin(theta_x)],
                                  [0, np.sin(theta_x), np.cos(theta_x)]])

    rotation_matrix_y = np.array([[np.cos(theta_y), 0, np.sin(theta_y)],
                                  [0, 1, 0],
                                  [-np.sin(theta_y), 0, np.cos(theta_y)]])

    combined_rotation_matrix = np.dot(rotation_matrix_y, rotation_matrix_x)
    rotated_points = np.dot(all_points, combined_rotation_matrix.T)
    return rotated_points


def is_point_in_triangle_2d(vertices, point):
    """
    After dimension reduction, we check if our intersection point with the plane of the triangle (the one of a segment
    and a plane) is inside the triangle
    """#
    v1, v2, v3 = vertices
    first_orientation = find_orientation(v1,v2,point)
    second_orientation = find_orientation(v2,v3,point)
    third_orientation = find_orientation(v3,v1, point)
    if first_orientation==0 and on_line_segment(v1,v2,point):
        return True
    if second_orientation==0 and on_line_segment(v2,v3,point):
        return True
    if third_orientation==0 and on_line_segment(v3,v1, point):
        return True
    return first_orientation==second_orientation and second_orientation==third_orientation


def get_segment_and_plane_intersection_3D(vertices,segment):
    """
    getting the intersection point or segment of a segment and a plane. Returns None if they don't intersect
    """#
    v1,v2,v3 = vertices
    vec12 = v2-v1
    vec13 = v3-v1
    normal = np.cross(vec12,vec13) # our normal vector
    A,B,C = normal
    D = -(A*v1[0]+B*v1[1]+C*v1[2])

    x1,y1,z1 = segment[0]
    x2,y2,z2 = segment[1]

    seg_vec_x =x2-x1
    seg_vec_y =y2-y1
    seg_vec_z =z2-z1

    seg_vec = np.array([seg_vec_x,seg_vec_y,seg_vec_z])



    if abs(np.dot(seg_vec,normal))< 1e-10: ## the normal to plane and vec sec are perpendicular
        # meaning the segment is either parallel or inscribed in the plane.
        if abs(A*x1+B*y1+C*z1+D)>1e-10: # if it's only parallel
            return None
        else: # it is inscribed in the plane with the triangle I return to 2d and check for intersection
            #of the segment with one of the other triangle's segments - so I return the segment endpoints.
            return segment # HERE

    # some mathematical explanation of the remainder of the function:
    # after we build the parametric equasion of our segment with seg_vec and one of the points for some parameter t
    # we insert the x, y , and z of it into our equasion of the plane (with parameters A,B,C, and D we found
    ## above) and then isolate our paameter t. if 0<=t<=1 then the segment intersects with the plane.

    #after insertion and isolation we get:

    t = (-D-A*x1-B*y1-C*z1)/(A*seg_vec_x+B*seg_vec_y+C*seg_vec_z)

    if 0<=t<=1:
        return (x1+seg_vec_x*t,y1+seg_vec_y*t,z1+seg_vec_z*t)
    else:
        return None

def do_segment_and_triangle_intersect_3D(vertices, segment):
    """
    1. I need to find the very point of intersection of the segment and a plane of the triangle
    2. once I have the point of intersection, I can find the rotation matrix into xy plane and turn it into 2D problem
    3. return whether in 2d the intersection point/segment is in the triangle
    """#
    point = get_segment_and_plane_intersection_3D(vertices, segment)
    if (point is None): #no intersection
        return False
    elif(len(point)==2): # the segment is inscribed in the plane
        if (vertices[0][2] != 0 or vertices[1][2] != 0 or vertices[2][2] != 0):
            rotated_vertex_and_point = rotate_to_same_z(vertices, point[0])
            ## and what else here to indent and what to do
            other_end_point = rotate_to_same_z(vertices, point[1])[-1]
            other_end_point = other_end_point.reshape(1,-1)
            rotated_vertex_and_point = np.concatenate((rotated_vertex_and_point, other_end_point))
        else:  #when rotation is unnecessary
            point = np.array(point)
            point = point.reshape(2, -1)
            rotated_vertex_and_point = np.concatenate((vertices, point))  # NEW
        assert (len(rotated_vertex_and_point) == 5)
        twoD_points = rotated_vertex_and_point[:, :2]
        for i in range(len(twoD_points)): # numeric stability reasons
            for j in range(len(twoD_points[0])):  # numeric stability reasons
                if np.abs(twoD_points[i][j]) < 1e-10:
                    twoD_points[i][j]=0
        assert (len(twoD_points)==5)
        vertices,segment = twoD_points[0:3],twoD_points[3:]
        v_segments = np.array([[vertices[0], vertices[1]], [vertices[1], vertices[2]], [vertices[2], vertices[0]]])
        for i in range(len(v_segments)): # checking intersection of segments
            if (segments_intersect(v_segments[i][0],v_segments[i][1],segment[0],segment[1])):
                return True
        return is_point_in_triangle_2d(twoD_points[0:3], twoD_points[-2]) or\
        is_point_in_triangle_2d(twoD_points[0:3], twoD_points[-1])
        # checking whether a segmengt is strictly inscribed in a triangle

    else: # single modest intersection point of the segment with the plane
        if (vertices[0][2]!=0 or vertices[1][2]!=0 or vertices[2][2]!=0): # if it's not already parallel to xy-plane
            rotated_vertex_and_point = rotate_to_same_z(vertices, point)
        else: # no rotation needed
            point = np.array(point)
            point = point.reshape(1,-1)
            rotated_vertex_and_point = np.concatenate((vertices,point))
        assert (len(rotated_vertex_and_point) == 4)
        twoD_points = rotated_vertex_and_point[:, :2]
        for i in range(len(twoD_points)): # numeric stability reasons
            for j in range(len(twoD_points[0])):  # numeric stability reasons
                if np.abs(twoD_points[i][j]) < 1e-10:
                    twoD_points[i][j]=0

        return is_point_in_triangle_2d(twoD_points[0:3],twoD_points[-1])
        # using the 2D vertices of triangle and intersection point


def two_triangles_intersect_3D(two_triangle):
    """
    primary function for triangles intersection in 3D
    utilizes key observation: Two triangles intersect iff at least one segment of one triangle intersect with 
    the other triangle
    """#
    vertices1 =np.array(two_triangle[0])
    vertices2 =np.array(two_triangle[1])
    segments1 = np.array([[vertices1[0], vertices1[1]], [vertices1[1], vertices1[2]], [vertices1[2], vertices1[0]]])
    segments2 = np.array([[vertices2[0], vertices2[1]], [vertices2[1], vertices2[2]], [vertices2[2], vertices2[0]]])

    ## basically going over each combination of segment and the other triangle
    for i in range(len(segments1)):
        segment = segments1[i]
        if (do_segment_and_triangle_intersect_3D(vertices2,segment)):
            return True

    for i in range(len(segments2)):
        segment = segments2[i]
        if (do_segment_and_triangle_intersect_3D(vertices1,segment)):
            return True
    return False

def primary_function(list_of_tuples):
    """
    Assuming vertices of each triangle are not colinear.
    """#
    if(len(list_of_tuples[0][0])==2):
        return two_2D_triangles_intersect(list_of_tuples[0], list_of_tuples[1])
    return two_triangles_intersect_3D(list_of_tuples)


