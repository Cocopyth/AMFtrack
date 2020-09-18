from scipy import sparse
def dilate(sparse_matrix_doc):
    dilated = sparse.dok_matrix(sparse_matrix_doc.shape, dtype=bool)
    for pos in sparse_matrix_doc.keys():
        positions=[pos,(pos[0],pos[1]+1),(pos[0],pos[1]-1),(pos[0]-1,pos[1]-1),(pos[0]+1,pos[1]-1),(pos[0]+1,pos[1]+1),(pos[0]-1,pos[1]+1),(pos[0]-1,pos[1]),(pos[0]+1,pos[1])]
        for position in positions:
            if position[0]<sparse_matrix_doc.shape[0] and position[1]<sparse_matrix_doc.shape[1] and position[0]>=0 and position[0]>=0:
                dilated[position]=1
    return(dilated)


def zeroToOne(thin_image,i,j):
    p2 = thin_image[i-1,j-1]
    p3 = thin_image[i-1,j]
    p4 = thin_image[i-1,j+1]
    p5 = thin_image[i,j+1]
    p6 = thin_image[i+1,j+1]
    p7 = thin_image[i+1,j]
    p8 = thin_image[i+1,j-1]
    p9 = thin_image[i,j-1]
    count = 0;
    endpoint = 0;
    if(p2+p3+p4+p5+p6+p7+p8+p9 == 1):
        endpoint = 1
    if(p2==0 and p3==1):
        count = count + 1
    if(p3==0 and p4==1):
        count = count + 1
    if(p4==0 and p5==1):
        count = count + 1
    if(p5==0 and p6==1):
        count = count + 1
    if(p6==0 and p7==1):
        count = count + 1
    if(p7==0 and p8==1):
        count = count + 1
    if(p8==0 and p9==1):
        count = count + 1
    if(p9==0 and p2==1):
        count = count + 1
    return count,endpoint




def stentiford(image):
    # Make copy of the image so that original image is not lost
    thin_image=sparse.dok_matrix((image.shape[0]+2,image.shape[1]+2), dtype=bool)
    thin_image[1:image.shape[0]+1,1:image.shape[1]+1]=image.copy()
    check = 2
    template = 1
    outImage = 1
    # Perform iterations as long as there are pixels marked for deletion
    iteration = 0
    total_changes = 0
    while(outImage):
        # Make outImage empty
        outImage = []
        changes = 0
        iteration = iteration + 1 
        # Loop through the pixels of the thin_image
        for pos in thin_image.keys():
            i=pos[0]
            j=pos[1]
            p0 = thin_image[i,j]
            p1 = thin_image[i-1,j]
            p2 = thin_image[i,j+1]
            p3 = thin_image[i+1,j]
            p4 = thin_image[i,j-1]
            if(template==1):
                template_match = (p1==0 and p3==1)
            if(template==2):
                template_match = (p2==1 and p4==0)
            if(template==3):
                template_match = (p1==1 and p3==0)
            if(template==4):
                template_match = (p2==0 and p4==1)
            connectivity, isEndpoint = zeroToOne(thin_image,i,j)
            if(template_match==1):
                if(connectivity == 1):
                    if(isEndpoint== 0):
                        outImage.append((i,j))


        # Delete the pixels marked for deletion
        for i,j in outImage:
            thin_image[i,j] = 0
            changes = changes + 1
        template = template+1
        if(template==5):
            template = 1
        print("iteration: ", iteration, "changes: ", changes)
        total_changes = total_changes + changes

    print("total_changes: ", total_changes)
    return thin_image

def hilditch(image):
    Image_Thinned=sparse.dok_matrix((image.shape[0]+2,image.shape[1]+2), dtype=bool)
    Image_Thinned[1:image.shape[0]+1,1:image.shape[1]+1]=image.copy()
    changing1 = 1
    i = 0
    while changing1:
        changes_occured = 0
        changing1 = []
        for pos in  Image_Thinned.keys():
            x=pos[0]
            y=pos[1]
            P2,P3,P4,P5,P6,P7,P8,P9 = n = neighbours(x, y, Image_Thinned)
            condition3 = (P4 * P6 * P8 == 0) or (zeroToOne(Image_Thinned,x-1,y-1)!=1)#p2
            condition4 = (P4 * P6 * P6 == 0) or (zeroToOne(Image_Thinned,x-1,y+1)!=1)#p4
            if (Image_Thinned[x,y] == 1 and 2 <= sum(n) <= 6 and transitions(n) == 1 and condition3 ==1 and condition4==1):
                    changing1.append((x,y))
        for x, y in changing1: 
            Image_Thinned[x,y] = 0
            changes_occured = changes_occured + 1

        i = i + 1
        print("Iteration: ", i , "changes_occured: ", changes_occured)
    return Image_Thinned


def neighbours(x,y,image):
    "Return 8-neighbours of image point P1(x,y), in a clockwise order"
    img = image
    x_1, y_1, x1, y1 = x-1, y-1, x+1, y+1
    return [ img[x_1,y], img[x_1,y1], img[x,y1], img[x1,y1],     # P2,P3,P4,P5
                img[x1,y], img[x1,y_1], img[x,y_1], img[x_1,y_1] ]    # P6,P7,P8,P9

def transitions(neighbours):
    "No. of 0,1 patterns (transitions from 0 to 1) in the ordered sequence"
    n = neighbours + neighbours[0:1]      # P2, P3, ... , P8, P9, P2
    return sum( (n1, n2) == (0, 1) for n1, n2 in zip(n, n[1:]) )  # (P2,P3), (P3,P4), ... , (P8,P9), (P9,P2)

def zhangSuen(image):
    "the Zhang-Suen Thinning Algorithm"
    Image_Thinned=sparse.dok_matrix((image.shape[0]+2,image.shape[1]+2), dtype=bool)
    for pos in  image.keys():
        Image_Thinned[pos[0]+1,pos[1]+1]=image[pos[0],pos[1]]
    changing1 = changing2 = 1        #  the points to be removed (set as 0)
    while changing1 or changing2:   #  iterates until no further changes occur in the image
        # Step 1
        changing1 = []
        for pos in  Image_Thinned.keys():
            x=pos[0]
            y=pos[1]
            P2,P3,P4,P5,P6,P7,P8,P9 = n = neighbours(x, y, Image_Thinned)
            if (Image_Thinned[x,y] == 1     and    # Condition 0: Point P1 in the object regions 
                2 <= sum(n) <= 6   and    # Condition 1: 2<= N(P1) <= 6
                transitions(n) == 1 and    # Condition 2: S(P1)=1  
                P2 * P4 * P6 == 0  and    # Condition 3   
                P4 * P6 * P8 == 0):         # Condition 4
                changing1.append((x,y))
        for x, y in changing1: 
            Image_Thinned[x,y] = 0
        # Step 2
        changing2 = []
        for pos in  Image_Thinned.keys():
            x=pos[0]
            y=pos[1]
            P2,P3,P4,P5,P6,P7,P8,P9 = n = neighbours(x, y, Image_Thinned)
            if (Image_Thinned[x,y] == 1   and        # Condition 0
                2 <= sum(n) <= 6  and       # Condition 1
                transitions(n) == 1 and      # Condition 2
                P2 * P4 * P8 == 0 and       # Condition 3
                P2 * P6 * P8 == 0):            # Condition 4
                changing2.append((x,y))    
        for x, y in changing2: 
            Image_Thinned[x,y] = 0
        print(len(changing1),len(changing2))
    return Image_Thinned