

############################################################################################
### Following functions are just a helper for programmers
# They show you how to for reversing the Ardent and the LM-based delauney registration
############################################################################################

def func_delauney_registration_reverse(section_savepath, atlas_img_path, section_img_path, source_landmark_points0, target_landmark_points0):
    #tempfolder = "C:/Users/q371ms/Downloads/delauney"

    target_img = cv.imread(section_img_path)  # Section image   ################
    source_img = cv.imread(atlas_img_path)  # Atlas image
    target_gray = cv.cvtColor(target_img, cv.COLOR_BGR2GRAY)
    source_gray = cv.cvtColor(source_img, cv.COLOR_BGR2GRAY)
    mask = np.zeros_like(target_gray)
    triangs_mask_src = np.zeros_like(source_img)
    triangs_mask_trg = np.zeros_like(target_img)
    height, width, channels = target_img.shape
    target_new_hull = np.zeros((height, width, channels), np.uint8)
    # target
    #source_landmark_points, target_landmark_points = convert_lm_points_scale(source_lms, target_lms)
    #source_landmark_points, target_landmark_points = target_lms, source_lms  ################
    #target_landmark_points = calculate_n_add_corner_points(target_gray, target_landmark_points)
    #source_landmark_points = calculate_n_add_corner_points(source_gray, source_landmark_points)  
    #source_landmark_points = calculate_n_add_corner_points(target_gray, source_landmark_points)  
    # print("lms targ rev ", target_landmark_points[0])
    # print("lms src rev ", source_landmark_points[0])
    # target_landmark_points0, source_landmark_points0 = source_landmark_points, target_landmark_points
    source_landmark_points, target_landmark_points = target_landmark_points0, source_landmark_points0
    """
    for point in target_landmark_points:
        co2, ro2 = point   # Level 3  c, r = xo1, yo1
        cv.circle(target_img, (co2, ro2), 3, (255, 255, 0), -1)
    cv.imwrite(os.path.join(section_savepath, f"rev_landmarks_target.png"), target_img)

    for point in source_landmark_points:
        co2, ro2 = point  # Level 3  c, r = xo1, yo1
        cv.circle(source_img, (co2, ro2), 3,  (255, 0, 255), -1)
    cv.imwrite(os.path.join(section_savepath, f"rev_landmarks_source.png"), source_img)
    """

    # print("lms targ rev ", target_landmark_points)
    # print("lms src rev ", source_landmark_points)
    points = np.array(target_landmark_points, np.int32)    
    convexhull = cv.convexHull(points)
    rect = cv.boundingRect(convexhull)
    save_target_lm_points = np.array(target_landmark_points, np.int32)
    # Delaunay triangulation
    subdiv = cv.Subdiv2D(rect)
    for p in target_landmark_points:
        subdiv.insert(p)
    triangles = subdiv.getTriangleList()
    triangles = np.array(triangles, dtype=np.int32)

    indexes_triangles = []
    for t in triangles:

        cv.polylines(triangs_mask_src,[t],True,(0,255,255))

        pt1 = (t[0], t[1])
        pt2 = (t[2], t[3])
        pt3 = (t[4], t[5])
        index_pt1 = np.where((points == pt1).all(axis=1))
        index_pt1 = extract_index_nparray(index_pt1)
        index_pt2 = np.where((points == pt2).all(axis=1))
        index_pt2 = extract_index_nparray(index_pt2)
        index_pt3 = np.where((points == pt3).all(axis=1))
        index_pt3 = extract_index_nparray(index_pt3)
        """"""
        #cv.line(target_img, pt1, pt2, (0,0,255))
        #cv.line(target_img, pt2, pt3, (0,0,255))
        #cv.line(target_img, pt3, pt1, (0,0,255))
        
        if index_pt1 is not None and index_pt2 is not None and index_pt3 is not None:
            triangle = [index_pt1, index_pt2, index_pt3]
            indexes_triangles.append(triangle)
    # source
    #cv.imwrite(os.path.join(tempfolder, 'target_triangles.png'),target_img)

    points2 = np.array(source_landmark_points, np.int32)
    convexhull2 = cv.convexHull(points2)
    rect2 = cv.boundingRect(convexhull2)

    save_source_lm_points = np.array(source_landmark_points, np.int32)

    # Triangulation of both faces
    transformation_functions = []
    triangles1 = []
    triangles2 = []
    iii = 1
    for triangle_index in indexes_triangles:
        # Triangulation of the first image: Source
        tr1_pt1 = source_landmark_points[triangle_index[0]]
        tr1_pt2 = source_landmark_points[triangle_index[1]]
        tr1_pt3 = source_landmark_points[triangle_index[2]]
        triangle1 = np.array([tr1_pt1, tr1_pt2, tr1_pt3], np.int32)
        triangles1.append(triangle1)
        rect1 = cv.boundingRect(triangle1)
        cv.polylines(triangs_mask_trg,[triangle1],True,(0,255,255))
        (x, y, w, h) = rect1
        cropped_triangle = source_img[y: y + h, x: x + w]
        cropped_tr1_mask = np.zeros((h, w), np.uint8)
        subpoints = np.array([[tr1_pt1[0] - x, tr1_pt1[1] - y],
                        [tr1_pt2[0] - x, tr1_pt2[1] - y],
                        [tr1_pt3[0] - x, tr1_pt3[1] - y]], np.int32)

        cv.fillConvexPoly(cropped_tr1_mask, subpoints, 255)

        # Lines space
        # Triangulation of second face
        tr2_pt1 = target_landmark_points[triangle_index[0]]
        tr2_pt2 = target_landmark_points[triangle_index[1]]
        tr2_pt3 = target_landmark_points[triangle_index[2]]
        triangle2 = np.array([tr2_pt1, tr2_pt2, tr2_pt3], np.int32)
        triangles2.append(triangle2)
        
        rect2 = cv.boundingRect(triangle2)
        (x, y, w, h) = rect2
        cropped_tr2_mask = np.zeros((h, w), np.uint8)
        subpoints2 = np.array([[tr2_pt1[0] - x, tr2_pt1[1] - y],
                            [tr2_pt2[0] - x, tr2_pt2[1] - y],
                            [tr2_pt3[0] - x, tr2_pt3[1] - y]], np.int32)
        cv.fillConvexPoly(cropped_tr2_mask, subpoints2, 255)
        # Warp triangles
        subpoints = np.float32(subpoints)
        subpoints2 = np.float32(subpoints2)
        M = cv.getAffineTransform(subpoints, subpoints2)
        transformation_functions.append(M)
        warped_triangle = cv.warpAffine(cropped_triangle, M, (w, h))
        warped_triangle = cv.bitwise_and(warped_triangle, warped_triangle, mask=cropped_tr2_mask)

        # Reconstructing destination face
        target_new_hull_rect_area = target_new_hull[y: y + h, x: x + w]
        target_new_hull_rect_area_gray = cv.cvtColor(target_new_hull_rect_area, cv.COLOR_BGR2GRAY)
        _, mask_triangles_designed = cv.threshold(target_new_hull_rect_area_gray, 1, 255, cv.THRESH_BINARY_INV)
        warped_triangle = cv.bitwise_and(warped_triangle, warped_triangle, mask=mask_triangles_designed)
        target_new_hull_rect_area = cv.add(target_new_hull_rect_area, warped_triangle)
        target_new_hull[y: y + h, x: x + w] = target_new_hull_rect_area
        iii+=1
    #("number of triangles: ", triangles1, triangles2)
    # Face swapped (putting 1st face into 2nd face)
    target_hull_mask = np.zeros_like(target_gray)
    target_inv_hull_mask = cv.fillConvexPoly(target_hull_mask, convexhull2, 255)
    target_hull_mask = cv.bitwise_not(target_inv_hull_mask)
    target_nohull = cv.bitwise_and(target_img, target_img, mask=target_hull_mask)
    rev_registered_img = cv.add(target_nohull, target_new_hull)
    #rev_registered_img_path = os.path.join(section_savepath, 'rev_delauney.png')
    #cv.imwrite(os.path.join(section_savepath, 'rev_delauney.png'), rev_registered_img)

    cv.imwrite(os.path.join(section_savepath, 'triangs_mask_trg.png'), triangs_mask_trg)
    cv.imwrite(os.path.join(section_savepath, 'triangs_mask_src.png'), triangs_mask_src)
    return rev_registered_img


def func_ardent_registration_reverse(sectionpath, source_img_path):
    source_gr = cv.imread(os.path.join(sectionpath, "source_img.png"),0) #section image
    # print(source_gr.shape)
    source_rgb = cv.imread(source_img_path) #unlabeled atlas
    b,g,r = cv.split(source_rgb)
    deformed_source_b = transform.transform_image(subject=b,
        subject_resolution=target_resolution,
        output_shape=source_gr.shape, deform_to='template')
    deformed_source_g = transform.transform_image( subject=g,
        subject_resolution=target_resolution,
        output_shape=source_gr.shape, deform_to='template')
    deformed_source_r = transform.transform_image(subject=r,
        subject_resolution=target_resolution,
        output_shape=source_gr.shape, deform_to='template')
    
    deformed_source_rgb = cv.merge([deformed_source_b, deformed_source_g, deformed_source_r])
    deformed_source_rgb = deformed_source_rgb.astype(np.uint8)
    #cv.imwrite(os.path.join(sectionpath, "rev_ardent.png"), deformed_source_rgb)
    return deformed_source_rgb

def func_ambia_registration_reverse(reg_code_status, sectionpath):

    target_img_path = os.path.join(sectionpath, "final_registered.png")
    source_img_path = os.path.join(sectionpath, "atlas_unlabeled_low.png")
    source_img_path2 = os.path.join(sectionpath, "atlas_unlabeled.png")
    registered_section_img_path =  os.path.join(sectionpath, "atlas_unlabeled.png")
    registered_section_img = cv.imread(registered_section_img_path)
    if reg_code_status == "ardent_reg":
        registered_section_img = func_ardent_registration_reverse(sectionpath, source_img_path)
    
    elif reg_code_status == "delauney_reg":
        target_img_path = os.path.join(sectionpath, "source_img.png")
        registered_section_img = func_delauney_registration_reverse(sectionpath, source_img_path, target_img_path, source_landmark_points, target_landmark_points)
        cv.imwrite(os.path.join(sectionpath, "rev_delauney_atlas_low.png"), registered_section_img)

        
        registered_section_img2 = func_delauney_registration_reverse(sectionpath, source_img_path2, target_img_path, source_landmark_points, target_landmark_points)
        cv.imwrite(os.path.join(sectionpath, "rev_delauney_atlas.png"), registered_section_img2)

    elif reg_code_status == "ardent_delauney_reg": 
        delauney_rev_img = func_delauney_registration_reverse(sectionpath, source_img_path, target_img_path, source_landmark_points, target_landmark_points)
        delauney_rev_img_path = os.path.join(sectionpath, 'rev_delauneyard_atlas_low.png')
        cv.imwrite(delauney_rev_img_path, delauney_rev_img)
        registered_section_img = func_ardent_registration_reverse(sectionpath, delauney_rev_img_path)

        source_img_path2= source_img_path[:-8]+source_img_path[-4:]
        delauney_rev_img2 = func_delauney_registration_reverse(sectionpath, source_img_path2, target_img_path, source_landmark_points, target_landmark_points)
        delauney_rev_img_path2 = os.path.join(sectionpath, 'rev_delauneyard_atlas.png')
        cv.imwrite(delauney_rev_img_path2, delauney_rev_img2)
        ardent_reg_section_img2 = func_ardent_registration_reverse(sectionpath, delauney_rev_img_path2)

        
        #ardent_reg_section_img_path = os.path.join(sectionpath, "rev_ardent.png")
        cv.imwrite(os.path.join(sectionpath, "rev_ardent_atlas.png"), ardent_reg_section_img2)

    print("reg_code_status ", reg_code_status)
    cv.imwrite(os.path.join(sectionpath, "rev_final_registered.png"), registered_section_img)

    return 
