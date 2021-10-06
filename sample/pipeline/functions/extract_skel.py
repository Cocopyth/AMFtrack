def extract_skel_tip_ext(im):
    im_cropped = im
    im_blurred =cv2.blur(im_cropped, (200, 200))
    im_back_rem = (im_cropped)/((im_blurred==0)*np.ones(im_blurred.shape)+im_blurred)*120
    im_back_rem[im_back_rem>=130]=130
    # # im_back_rem = im_cropped*1.0
    # # # im_back_rem = cv2.normalize(im_back_rem, None, 0, 255, cv2.NORM_MINMAX)
    frangised = frangi(im_back_rem,sigmas=range(1,20,4))*255
    # # frangised = cv2.normalize(frangised, None, 0, 255, cv2.NORM_MINMAX)
    hessian = hessian_matrix_det(im_back_rem,sigma = 20)
    blur_hessian = cv2.blur(abs(hessian), (20, 20))
#     transformed = (frangised+cv2.normalize(blur_hessian, None, 0, 255, cv2.NORM_MINMAX)-im_back_rem+120)*(im_blurred>=35)
#     transformed = (frangised+cv2.normalize(abs(hessian), None, 0, 255, cv2.NORM_MINMAX)-im_back_rem+120)*(im_blurred>=35)
    transformed = (frangised-im_back_rem+120)*(im_blurred>=35)
#     low = 20
#     high = 100
    lowt = (transformed > low).astype(int)
    hight = (transformed > high).astype(int)
    hyst = filters.apply_hysteresis_threshold(transformed, low, high)
    kernel = np.ones((3,3),np.uint8)
    dilation = cv2.dilate(hyst.astype(np.uint8) * 255,kernel,iterations = 1)
    for i in range(3):
        dilation=cv2.erode(dilation.astype(np.uint8) * 255,kernel,iterations = 1)
        dilation = cv2.dilate(dilation.astype(np.uint8) * 255,kernel,iterations = 1)
    dilated = dilation>0

    nb_components, output, stats, centroids = cv2.connectedComponentsWithStats(dilated.astype(np.uint8), connectivity=8)
    #connectedComponentswithStats yields every seperated component with information on each of them, such as size
    #the following part is just taking out the background which is also considered a component, but most of the time we don't want that.
    sizes = stats[1:, -1]; nb_components = nb_components - 1

    # minimum size of particles we want to keep (number of pixels)
    #here, it's a fixed value, but you can set it as you want, eg the mean of the sizes or whatever
    min_size = 4000  

    #your answer image
    img2 = np.zeros((dilated.shape))
    #for every component in the image, you keep it only if it's above min_size
    for i in range(0, nb_components):
        if sizes[i] >= min_size:
            img2[output == i + 1] = 1
    skeletonized = cv2.ximgproc.thinning(np.array(255*img2,dtype=np.uint8))
    nx_g = generate_nx_graph(from_sparse_to_graph(scipy.sparse.dok_matrix(skeletonized)))
    g,pos= nx_g
    tips = [node for node in g.nodes if g.degree(node)==1]
    dilated_bis = np.copy(img2)
    for tip in tips:
        branch = np.array(orient(g.get_edge_data(*list(g.edges(tip))[0])['pixel_list'],pos[tip]))
        orientation = branch[0]-branch[min(branch.shape[0]-1,20)]
        orientation = orientation/(np.linalg.norm(orientation))
        window = 20
        x,y = pos[tip][0],pos[tip][1]
        if x-window>=0 and x+window< dilated.shape[0] and y-window>=0 and y+window< dilated.shape[1]:
            shape_tip = dilated[x-window:x+window,y-window:y+window]
#             dist = 20
            for i in range(dist):
                pixel = (pos[tip]+orientation*i).astype(int)
                xp,yp = pixel[0],pixel[1]
                if xp-window>=0 and xp+window< dilated.shape[0] and yp-window>=0 and yp+window< dilated.shape[1]:
                    dilated_bis[xp-window:xp+window,yp-window:yp+window]+=shape_tip
    dilation = cv2.dilate(dilated_bis.astype(np.uint8) * 255,kernel,iterations = 1)
    for i in range(3):
        dilation=cv2.erode(dilation.astype(np.uint8) * 255,kernel,iterations = 1)
        dilation = cv2.dilate(dilation.astype(np.uint8) * 255,kernel,iterations = 1)
        return(dilation>0)