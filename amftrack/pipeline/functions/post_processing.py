def find_center_orth(directory,row):
    directory = directory_project
    directory_name = row['folder']
    path_snap=directory+directory_name
    path_tile=path_snap+'/Img/TileConfiguration.txt.registered'
    try:
        tileconfig = pd.read_table(path_tile,sep=';',skiprows=4,header=None,converters={2 : ast.literal_eval},skipinitialspace=True)
    except:
        print('error_name')
        path_tile=path_snap+'/Img/TileConfiguration.registered.txt'
        tileconfig = pd.read_table(path_tile,sep=';',skiprows=4,header=None,converters={2 : ast.literal_eval},skipinitialspace=True)
    dirName = path_snap+'/Analysis'
    shape = (3000,4096)
    try:
        os.mkdir(path_snap+'/Analysis') 
        print("Directory " , dirName ,  " Created ")
    except FileExistsError:
        print("Directory " , dirName ,  " already exists")  
    t=time()
    xs =[c[0] for c in tileconfig[2]]
    ys =[c[1] for c in tileconfig[2]]
    dim = (int(np.max(ys)-np.min(ys))+4096,int(np.max(xs)-np.min(xs))+4096)
    ims = []
    for name in tileconfig[0]:
        imname = '/Img/'+name.split('/')[-1]
    #     ims.append(imageio.imread('//sun.amolf.nl/shimizu-data/home-folder/oyartegalvez/Drive_AMFtopology/PRINCE'+date_plate+plate_str+'/Img/'+name))
        ims.append(imageio.imread(directory+directory_name+imname))
    # contour = scipy.sparse.lil_matrix(dim,dtype=np.uint8)
    # half_circle = scipy.sparse.lil_matrix(dim,dtype=np.uint8)
    contour2 = scipy.sparse.lil_matrix((dim[0]//5,dim[1]//5),dtype=np.uint8)
    half_circle2 = scipy.sparse.lil_matrix((dim[0]//5,dim[1]//5),dtype=np.uint8)
    for index,im in enumerate(ims):
        im_cropped = im
        im_blurred =cv2.blur(im_cropped, (200, 200))
        im_back_rem = (im_cropped+1)/(im_blurred+1)*120
        im_back_rem[im_back_rem>=130]=130
    #     laplacian = cv2.Laplacian((im_cropped<=15).astype(np.uint8),cv2.CV_64F)
    #     points = laplacian>=4
    #     np.save(f'Temp\dilated{tileconfig[0][i]}',dilated)
        boundaries = int(tileconfig[2][index][0]-np.min(xs)),int(tileconfig[2][index][1]-np.min(ys))
    #     contour[boundaries[1]:boundaries[1]+shape[0],boundaries[0]:boundaries[0]+shape[1]] += points
        contour2[boundaries[1]//5:boundaries[1]//5+shape[0]//5,boundaries[0]//5:boundaries[0]//5+shape[1]//5] += cv2.resize((im_blurred<=20).astype(np.uint8),(im.shape[1]//5,im.shape[0]//5))
    #     if index<=80:
    #         half_circle[boundaries[1]:boundaries[1]+shape[0],boundaries[0]:boundaries[0]+shape[1]] += points
    pivot = 3000
    circle = []
    border = []
    for x in range(2000,8000):
        indexes = np.where(contour2[:,x].toarray()>0)[0]
        distances = indexes-pivot
        X = list(indexes)
        Y = list(distances)
        sort_indexes = [x for _, x in sorted(zip(Y, X), key=lambda pair: 1/pair[0])]
        candidates = sort_indexes[0],sort_indexes[-1]
        circle.append((x,min(candidates)))
        border.append((x,max(candidates)))
    array_circ = np.zeros(contour2.shape)
    array_bord = np.zeros(contour2.shape)
    for point in circle:
        array_circ[point[1],point[0]]=1
    for point in border:
        array_bord[point[1],point[0]]=1
    x = np.array(border)[:,1]
    y = np.array(border)[:,0].reshape(-1, 1)

    reg = LinearRegression().fit(y, x)
    orthog = [-1/reg.coef_[0],1]
    orthog = np.array(orthog)/np.linalg.norm(orthog)
    circle_correct=[point for point in circle if point[1]<=3000]
    x = np.array(circle_correct)[:,1]
    y = np.array(circle_correct)[:,0]
    x_m = np.mean(x)
    y_m = np.mean(y)
    method_2 = "leastsq"

    def calc_R(xc, yc):
        """ calculate the distance of each 2D points from the center (xc, yc) """
        return np.sqrt((x-xc)**2 + (y-yc)**2)

    def f_2(c):
        """ calculate the algebraic distance between the data points and the mean circle centered at c=(xc, yc) """
        Ri = calc_R(*c)
        return Ri - Ri.mean()

    center_estimate = x_m, y_m
    center_2, ier = optimize.leastsq(f_2, center_estimate)

    xc_2, yc_2 = center_2
    Ri_2       = calc_R(*center_2)
    R_2        = Ri_2.mean()
    residu_2   = sum((Ri_2 - R_2)**2)
    return((xc_2, yc_2),orthog)