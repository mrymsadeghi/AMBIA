
import os
import cv2 as cv
import ardent
import numpy as np
import Switches_Static as st_switches
from Switches_Dynamic import set_reg_code
import time
from GuiFunctions import save_to_saved_data_pickle
ACC_DF = st_switches.ACC_DF                  #Accelarating factor for ardent registration. Downscales both images by this factor to speed up the registration process
TEMP_DF = st_switches.TEMP_DF
TARG_DF = st_switches.TARG_DF
source_resolution = st_switches.source_resolution
target_resolution = st_switches.target_resolution
registered_img_path = ''
target_show_img_path = ''


def transform_target_img(target_img_path):
    image = cv.imread(target_img_path)
    image2 = image.copy()
    image2[np.all(image != [0,0,0], axis=-1)] = (49, 165, 4)
    image2[np.all(image == [139,249,128], axis=-1)] = (0,0,0)
    image2[np.all(image == [46,103,136], axis=-1)] = (164,215,143)
    image2[np.all(image == [255,0,0], axis=-1)] = (49, 165, 4)
    return image2


def func_ardent_registration(section_savepath, source_img_path, target_img_path):
    time1 = time.time()
    """
    Inputs atlas img path, and section img path in alevel
    Dimentions of Section image and Atlas image are both adjusted to half the dimentions of atlas img
    atlas sizes are (800, 1140) , Downscaled by factor of 2 -> atlas.shape = (400, 570)
    alevel img shape e.g = (1372, 1872), when resized, section.shape = (400, 570)
    for accelarator factor ACC_DF = 1 , time apprx 40sec, for ACC_DF = 2, time apprx 9 sec
    """
    global transform
    global ardnt_src_shape, ardnt_targ_shape

    #target_rgb = cv.imread(target_img_path)
    source_rgb = cv.imread(source_img_path)

    target_rgb = transform_target_img(target_img_path)
    target = cv.cvtColor(target_rgb, cv.COLOR_BGR2GRAY)
    source = cv.cvtColor(source_rgb, cv.COLOR_BGR2GRAY)
    cv.imwrite(os.path.join(section_savepath, "dummy_target.png"), target_rgb)
    target_shape, source_shape = target.shape, source.shape

    ardnt_targ_shape, ardnt_src_shape = target.shape, source.shape

    dftt =  ((source_shape[0]*ACC_DF)/target_shape[0], (source_shape[1]*ACC_DF)/target_shape[1])
    source = source.astype(float)
    target = target.astype(float)

    source = ardent.basic_preprocessing(source)
    target = ardent.basic_preprocessing(target)
    deformative_stepsize, sigma_regularization, sigma_contrast = 1e5, 1e4, 2  
    contrast_order, affine_stepsize = 4, 0.5 
    initial_affine, fixed_affine_scale, initial_velocity_fields,  = None, None, None
    classify_and_weight_voxels, spatially_varying_contrast_map = False, False
    artifact_prior, background_prior = 0.2, 0.01
    multiscales = 8, 4
    num_iterations              = 50, 100
    num_affine_only_iterations  = 50, 25
    num_rigid_affine_iterations = 25, 0
    calibrate, show = True, True  
    track_progress_every_n = 10  

    transform = ardent.Transform()
    try:
        transform.register(
            target=target,
            template=source,
            target_resolution=target_resolution,
            template_resolution=source_resolution,
            multiscales=multiscales,
            preset=None,
            affine_stepsize=affine_stepsize,
            deformative_stepsize=deformative_stepsize,
            sigma_regularization=sigma_regularization,
            num_iterations=num_iterations,
            num_affine_only_iterations=num_affine_only_iterations,
            num_rigid_affine_iterations=num_rigid_affine_iterations,
            fixed_affine_scale=fixed_affine_scale,
            initial_affine=initial_affine,
            initial_velocity_fields=initial_velocity_fields,
            contrast_order=contrast_order,
            sigma_contrast=sigma_contrast,
            classify_and_weight_voxels=classify_and_weight_voxels,
            sigma_matching=None,
            artifact_prior=artifact_prior,
            background_prior=background_prior,
            spatially_varying_contrast_map=spatially_varying_contrast_map,
            calibrate=calibrate,
            track_progress_every_n=track_progress_every_n,
        )
    except Exception as e:
        print('Exception occurred', e);  
    else:
        if show:
            print("End of Parameter tuning")

    b,g,r = cv.split(source_rgb)
    deformed_source_b = transform.transform_image(subject=b,
        subject_resolution=source_resolution,
        output_shape=target.shape, deform_to='target')
    deformed_source_g = transform.transform_image( subject=g,
        subject_resolution=source_resolution,
        output_shape=target.shape, deform_to='target')
    deformed_source_r = transform.transform_image(subject=r,
        subject_resolution=source_resolution,
        output_shape=target.shape, deform_to='target')
    
    deformed_source_rgb = cv.merge([deformed_source_b, deformed_source_g, deformed_source_r])
    deformed_source_rgb = deformed_source_rgb.astype(np.uint8)
    ardent_reg_img_path = os.path.join(section_savepath, "ardent_reg_img.png")
    cv.imwrite(ardent_reg_img_path, deformed_source_rgb)
    time2 = time.time()
    print(f"Registration took {str(round(time2-time1,1))} seconds")
    return deformed_source_rgb, ardent_reg_img_path


def calculate_n_add_corner_points(img_gray, reg_code):
    """ This function adds 20 landmarks automatically to the source and target images, to anchor the outer structure
    Since automatic registration using ardent almost always registers the outer structure well.
    """
    height, width = img_gray.shape
    _, img_mask = cv.threshold(img_gray, 5, 255, cv.THRESH_BINARY) 
    kernel = np.ones((5, 5), np.uint8)
    dilated_img_mask = cv.dilate(img_mask, kernel, iterations=1)

    contours, _ = cv.findContours(dilated_img_mask, mode=cv.RETR_TREE, method=cv.CHAIN_APPROX_NONE,offset=(0, 0))
    biggest_cnt = max(contours, key = cv.contourArea)
    x,y,w,h = cv.boundingRect(biggest_cnt)

    minr, maxr, minc, maxc = int(y/2), y+h+int((height-(y+h))/2), int(x/2), x+w+int((width-(x+w))/2)

    r_linespace = np.linspace(minr, maxr, num = st_switches.Delauney_strength)
    c_linespace = np.linspace(minc, maxc, num = st_switches.Delauney_strength)

    anchor_lms = []
    rr1 = 1
    rr2 = -2

    anchor_lms.append((int(c_linespace[0]), int(r_linespace[rr1])))
    anchor_lms.append((int(c_linespace[0]), int(r_linespace[rr2])))
    anchor_lms.append((int(c_linespace[-1]), int(r_linespace[rr1])))
    anchor_lms.append((int(c_linespace[-1]), int(r_linespace[rr2])))

    anchor_lms.append((int(c_linespace[rr1]), int(r_linespace[0])))
    anchor_lms.append((int(c_linespace[rr2]), int(r_linespace[0])))
    anchor_lms.append((int(c_linespace[rr1]), int(r_linespace[-1])))
    anchor_lms.append((int(c_linespace[rr2]), int(r_linespace[-1])))

    if reg_code == "ardent_reg":
        nr_of_points = 20
        intrval = int(len(biggest_cnt)/nr_of_points)
        lm_list = list(range(0, len(biggest_cnt), intrval))
        for ipnt in lm_list:
            cval,rval = biggest_cnt[ipnt][0][0],biggest_cnt[ipnt][0][1]
            anchor_lms.append((int(cval), int(rval)))

    return anchor_lms


def extract_index_nparray(nparray):
    index = None
    for num in nparray[0]:
        index = num
        break
    return index


def convert_lm_points_scale(source_lms, target_lms):
    converted_source_lm_points = [(int(coords[0]/(TEMP_DF*ACC_DF)),int(coords[1]/(TEMP_DF*ACC_DF))) for coords in source_lms if coords] 
    converted_target_lm_points = [(int(coords[0]/(TARG_DF*ACC_DF)),int(coords[1]/(TARG_DF*ACC_DF))) for coords in target_lms if coords] 
    return converted_source_lm_points, converted_target_lm_points


def func_delauney_registration(section_savepath, source_lms, target_lms, source_img_path, target_img_path, reg_code, section):
    """ 
      Inputs img1coords and img2coords  are list of Landmark coords (c,r)
      these are only landmarks chosen by the user manually through GUI
      atlas_landmark_points and tissue_landmark_points   are list of Landmark coords (c,r)
      these are landmarks chosen manually by the user plus the corner landmarks
      both lists img1coords and atlas_landmark_points, are in the scale of atlas image 
      downscaled by factor of two
      Always first 4 coords of atlas/tissue_landmark_points are the corner coords
      Saves atlas/tissue_landmark_points as npy
    """
    global triangles1
    global triangles2
    global transformation_functions
    global target_landmark_points
    global source_landmark_points
    target_img = cv.imread(target_img_path)  # Atlas image   ################
    source_img = cv.imread(source_img_path)  # Section image
    target_gray = cv.cvtColor(target_img, cv.COLOR_BGR2GRAY)
    source_gray = cv.cvtColor(source_img, cv.COLOR_BGR2GRAY)
    source_img_show = source_img.copy()
    target_img_show = target_img.copy()

    target_new_hull = np.zeros_like(target_img, np.uint8)
    # target
    #source_landmark_points, target_landmark_points = convert_lm_points_scale(source_lms, target_lms)
    source_landmark_points0, target_landmark_points0 = source_lms, target_lms    ################

    target_anchor_lms = calculate_n_add_corner_points(target_gray, reg_code)
    source_anchor_lms = target_anchor_lms.copy()
    if reg_code != "ardent_reg":
        source_anchor_lms = calculate_n_add_corner_points(source_gray, reg_code) 

    
    

    target_landmark_points = target_anchor_lms + target_landmark_points0
    source_landmark_points = source_anchor_lms + source_landmark_points0
    #source_landmark_points = calculate_n_add_corner_points(target_gray, source_landmark_points)  
    """
    for point in source_landmark_points:
        co2, ro2 = point  # Level 3  c, r = xo1, yo1
        cv.circle(source_img_show, (co2, ro2), 3, (255, 255, 0), -1)
    cv.imwrite(os.path.join(section_savepath, f"landmarks_source2.png"), source_img_show)

    for point in target_landmark_points:
        co2, ro2 = point  # Level 3  c, r = xo1, yo1
        cv.circle(target_img_show, (co2, ro2), 3,  (255, 0, 255), -1)
    cv.imwrite(os.path.join(section_savepath, f"landmarks_target2.png"), target_img_show)
    """
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
        pt1 = (t[0], t[1])
        pt2 = (t[2], t[3])
        pt3 = (t[4], t[5])
        index_pt1 = np.where((points == pt1).all(axis=1))
        index_pt1 = extract_index_nparray(index_pt1)
        index_pt2 = np.where((points == pt2).all(axis=1))
        index_pt2 = extract_index_nparray(index_pt2)
        index_pt3 = np.where((points == pt3).all(axis=1))
        index_pt3 = extract_index_nparray(index_pt3)
        
        #cv.line(target_img, pt1, pt2, (0,0,255))
        #cv.line(target_img, pt2, pt3, (0,0,255))
        #cv.line(target_img, pt3, pt1, (0,0,255))
        
        if index_pt1 is not None and index_pt2 is not None and index_pt3 is not None:
            triangle = [index_pt1, index_pt2, index_pt3]
            indexes_triangles.append(triangle)

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
    #registered_img = cv.add(target_nohull, target_new_hull)
    registered_img = target_new_hull
    #cv.imwrite(os.path.join(section_savepath, 'target_nohull.png'), target_nohull)
    landmarks_coords = {'source': save_source_lm_points, 'target': save_target_lm_points}
    #save_to_pkl("landmarks_coords.pkl", landmarks_coords)
    save_to_saved_data_pickle(landmarks_coords, 'landmarks_coords', section)
    delauney_reg_img_path = os.path.join(section_savepath, 'reg_delauney_img.png')
    cv.imwrite(delauney_reg_img_path, registered_img)

    return registered_img, delauney_reg_img_path


def resize_images_for_registration(source_img_path,labeled_atlas_LM_filepath,unlabeled_atlas_LM_filepath, sectionpath, general_unlabeled_atlas_filepath):
    
    target_rgb = cv.imread(unlabeled_atlas_LM_filepath)
    target_rgb2 = cv.imread(labeled_atlas_LM_filepath)
    source_rgb = cv.imread(source_img_path)
    target_rgb3 = cv.imread(general_unlabeled_atlas_filepath)
    target_img_path = os.path.join(sectionpath, "atlas_unlabeled.png")
    target_img_path2 = os.path.join(sectionpath, "atlas_labeled.png")
    target_img_path3 = os.path.join(sectionpath, "atlas_unlabeled_low.png")

    source_img_path = os.path.join(sectionpath, "source_img.png")
    if True:
        target_shape, source_shape = target_rgb.shape, source_rgb.shape
        source_rgb_resized = cv.resize(source_rgb, (int(source_shape[1]/(TEMP_DF*ACC_DF)), int(source_shape[0]/(TEMP_DF*ACC_DF))))
        target_rgb_resized = cv.resize(target_rgb, (int(target_shape[1]/(TARG_DF*ACC_DF)), int(target_shape[0]/(TARG_DF*ACC_DF))))
        target_rgb_resized2 = cv.resize(target_rgb2, (int(target_shape[1]/(TARG_DF*ACC_DF)), int(target_shape[0]/(TARG_DF*ACC_DF))))
        target_rgb_resized3 = cv.resize(target_rgb3, (int(target_shape[1]/(TARG_DF*ACC_DF)), int(target_shape[0]/(TARG_DF*ACC_DF))))
        cv.imwrite(source_img_path, source_rgb_resized)
        cv.imwrite(target_img_path, target_rgb_resized)
        cv.imwrite(target_img_path2, target_rgb_resized2)
        cv.imwrite(target_img_path3, target_rgb_resized3)
    else:    #Fix: rename instead of read and write
        cv.imwrite(source_img_path, source_rgb)
        cv.imwrite(target_img_path, target_rgb)
        cv.imwrite(target_img_path2, target_rgb2)
        cv.imwrite(target_img_path3, target_rgb3)

    return source_img_path, target_img_path


def func_ambia_registration(sectionpath, source_img_path, target_img_path, source_lms0, target_lms0, Ardent_reg_done, target_show_img_path0, section, auto_reg_switch=False):
    
    global source_lms
    global target_lms
    global registered_img_path
    global target_show_img_path

    source_lms = source_lms0
    target_lms = target_lms0
    reg_code = "No_reg"
    registered_img_path = source_img_path
    ardent_reg_img_path = os.path.join(sectionpath, "ardent_reg_img.png")
    target_show_img_path = target_show_img_path0

    # if auto_reg_switch is on automatic ardent registration will be done
    if auto_reg_switch:
        reg_code = "ardent_reg"
        # Check if ardent registration was performed
        if not Ardent_reg_done: #ardent registration wasn't performed
            target_img_path_ard = os.path.join(sectionpath, "atlas_unlabeled.png")
            _, registered_img_path = func_ardent_registration(sectionpath, source_img_path, target_img_path_ard)
        else:
            registered_img_path = os.path.join(sectionpath, "ardent_reg_img.png")
    # Check to see if there are any LMs selected by the user (len of lms we know are equal befor this func was called in Main)
    if len(source_lms) + len(target_lms) !=0:   
        if reg_code == "No_reg":            
            _, registered_img_path = func_delauney_registration(sectionpath, source_lms, target_lms, source_img_path, target_img_path, reg_code, section)
            reg_code = "delauney_reg"
        elif reg_code == "ardent_reg":
            source_img_path = ardent_reg_img_path
            #source_lms, target_lms = convert_lm_points_scale(source_lms, target_lms)
            _, registered_img_path = func_delauney_registration(sectionpath, source_lms, target_lms, source_img_path, target_img_path, reg_code, section)
            reg_code = "ardent_delauney_reg"   
    elif len(source_lms) + len(target_lms) ==0 and not auto_reg_switch:
        registered_img_path = target_img_path

    # print(registered_img_path)
    # print(target_show_img_path)
    registered_img = cv.imread(registered_img_path)
    target_img = cv.imread(target_show_img_path)
    overlayed_registered_img = cv.addWeighted(registered_img, 0.6, target_img, 0.4, 0)
    overlayed_registered_img_path = os.path.join(sectionpath, "overlayed_registered_img.png")
    cv.imwrite(overlayed_registered_img_path, overlayed_registered_img)
    cv.imwrite(os.path.join(sectionpath, "final_registered.png"), registered_img)
    
    set_reg_code(reg_code)
    return overlayed_registered_img_path, reg_code






####################################################
## Functions related to transforming point coords
####################################################

def area(x1, y1, x2, y2, x3, y3):
    return abs((x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2)) / 2.0)


def isInside(x1, y1, x2, y2, x3, y3, x, y):
    A = area (x1, y1, x2, y2, x3, y3)
    A1 = area (x, y, x2, y2, x3, y3)
    A2 = area (x1, y1, x, y, x3, y3)
    A3 = area (x1, y1, x2, y2, x, y)
    if(A == A1 + A2 + A3):
        return True
    else:
        return False


def delauney_transform(xo1, yo1):
    xo2,yo2 = 10,10
    for i in range(len(triangles1)):
        x1, x2, x3= triangles1[i][0][0], triangles1[i][1][0], triangles1[i][2][0]
        y1, y2, y3= triangles1[i][0][1], triangles1[i][1][1], triangles1[i][2][1]
        if isInside(x1, y1, x2, y2, x3, y3, xo1, yo1):
            M = np.array(transformation_functions[i])
            XY = np.array([[triangles1[i][0][0]],[triangles1[i][0][1]],[1]])
            XYt = M.dot(XY)
            xdiff, ydiff = int(triangles2[i][0][0] - XYt[0]), int(triangles2[i][0][1] - XYt[1])
            XYo1 = np.array([[xo1],[yo1],[1]])
            XYo2 = M.dot(XYo1)
            xo2, yo2 = int(XYo2[0] + xdiff), int(XYo2[1] + ydiff)
    return xo2,yo2


def delauney_transform_points(blobs_coords):
    blobs_coords = [(delauney_transform(coords[0], coords[1])) for coords in blobs_coords]
    delauney_converted_blobs_coords = [(int(coords[0]*(TARG_DF*ACC_DF)),int(coords[1]* (TARG_DF*ACC_DF))) for coords in blobs_coords if coords] #(c,r)
    return delauney_converted_blobs_coords


def ardent_transform_points(blobs_coords, section):
    MARGIN, DFba, section_savepath = section.get_levels_n_factors()
    temp_ho, temp_wo = int(ardnt_src_shape[0]/2), int(ardnt_src_shape[1]/2)
    targ_ho, targ_wo = int(ardnt_targ_shape[0]/2), int(ardnt_targ_shape[1]/2)
    # Transform blob coords from blevel to alevel
    blobs_coords0 = [(int(coords[0] / DFba) + MARGIN, int(coords[1] / DFba) + MARGIN) for coords in blobs_coords] #(c,r)

    res_df = 100
    blobs_coords1 = [(int(coords[0]/(TEMP_DF*ACC_DF)), int(coords[1] /(TEMP_DF*ACC_DF))) for coords in blobs_coords0] #(c,r)
    # print("blobs_coords1 ", blobs_coords1)
    #adjust coords for ardent
    #blobs_coords2 = [(int(((coords[1]-temp_ho)/target_resolution)*source_resolution), int(((coords[0]-temp_wo)/target_resolution)*source_resolution)) for coords in blobs_coords1] #(r,c)
    blobs_coords2 = [((coords[1]-temp_ho)*res_df, (coords[0]-temp_wo)*res_df) for coords in blobs_coords1] #(r,c)
    # print("blobs_coords2 ", blobs_coords2)
    # Apply Ardent Transformation
    blobs_coords3 = transform.transform_points(np.array(blobs_coords2), deform_to= 'target').tolist() 
    # print("blobs_coords3 ", blobs_coords3)
    blobs_coords3 = [coords for coords in blobs_coords3]  #(r,c)
    ## Transfer coords from position fields space to opencv
    blobs_coords4 = [((targ_ho+int(coords[0]/target_resolution)), (targ_wo+int(coords[1]/target_resolution))) for coords in blobs_coords3]  #(r,c)
    # print("blobs_coords4 ", blobs_coords4)
    blobs_coords5 = [(int(coords[1]),int(coords[0])) for coords in blobs_coords4 if coords] #(c,r)
    blobs_coords6 = [(int(coords[0]*(TARG_DF*ACC_DF)),int(coords[1]* (TARG_DF*ACC_DF))) for coords in blobs_coords5 if coords] #(c,r)
    ardent_converted_blobs_coords = blobs_coords6
    return blobs_coords5, ardent_converted_blobs_coords


def func_convert_coords(reg_code, blobs_coords, color, section):
    if reg_code == "ardent_reg":
        blobs_coords5, converted_blob_coords = ardent_transform_points(blobs_coords, section)
    elif reg_code == "delauney_reg":
        # Apply Delauney Transformation
        MARGIN, DFba, _ = section.get_levels_n_factors()
        blobs_coords0 = [(int(coords[0] / DFba) + MARGIN, int(coords[1] / DFba) + MARGIN) for coords in blobs_coords]
        blobs_coords1 = [(int(coords[0]/(TEMP_DF*ACC_DF)), int(coords[1] /(TEMP_DF*ACC_DF))) for coords in blobs_coords0] #(c,r)
        converted_blob_coords = delauney_transform_points(blobs_coords1)
    elif reg_code == "ardent_delauney_reg":
        blobs_coords5, ardent_converted_blob_coords = ardent_transform_points(blobs_coords, section)
        converted_blob_coords = delauney_transform_points(blobs_coords5)
    else:
        converted_blob_coords = blobs_coords
    return converted_blob_coords

def get_overlayed_registered_img(opacity_value, sectionpath):
    df_opacity_a = opacity_value/100
    df_opacity_b = 1 - df_opacity_a
    registered_img = cv.imread(registered_img_path)
    target_img = cv.imread(target_show_img_path)
    overlayed_registered_img = cv.addWeighted(registered_img, df_opacity_a, target_img, df_opacity_b, 0)
    overlayed_registered_img_path = os.path.join(sectionpath, "overlayed_registered_img.png")
    cv.imwrite(overlayed_registered_img_path, overlayed_registered_img)
    return overlayed_registered_img_path