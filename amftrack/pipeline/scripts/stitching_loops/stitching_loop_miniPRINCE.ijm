macro StitchingLoop40even{

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//Asking for the directory where are the folders with the images to be stitched
mainDirectory = "/projects/0/einf914/data/" ;

list = getFileList(mainDirectory);
//Stitching parameters
fileNames       = "Img_r{yy}_c{xx}.tif";

overlap         = "10";  

gridSizeX       = "10"
gridSizeY       = "14"
gridSizeZ       = "1"

startX          = "1"
startY          = "1"

fal             = "1";
rth             = "0.1";
mdt             = "6";
adt             = "6";
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////


/////////////////////////////////////////////////////////////////////////////////////////////////////////////////

for (i=0; i<=list.length-1; i=i+1){
	 if(startsWith(list[i],"20200701_1557_Plate13")) {
		outputDirectory = mainDirectory + File.separator + list[i]; 
        print(list[i]);
        inputDirectory  = outputDirectory + File.separator + "Img";
        inputDirectory2  = outputDirectory + File.separator + "Img2";
        File.makeDirectory(inputDirectory2);
        listImg = getFileList(inputDirectory);

        for (j=0; j<=listImg.length-1; j++){
            if(startsWith(listImg[j],"Img_r")){
                open(inputDirectory + File.separator + listImg[j]);
                run("Scale...", "x=0.5 y=0.5 interpolation=Bilinear average create");
                //run("Subtract Background...", "rolling=20 light");
                run("Window/Level...");
                setMinAndMax(75, 224);
                //run("Apply LUT");
                //run("Close");
                wait(100);
                saveAs("Tiff", inputDirectory2 + File.separator + listImg[j]);
                close();
            }
        }

        run("Stitch Sequence of Grids of Images", "grid_size_x=" + gridSizeX + " grid_size_y=" + gridSizeY + " grid_size_z=" + gridSizeZ + " overlap=" + overlap + " input=" + inputDirectory2 + " file_names=" + fileNames + " rgb_order=rgb output_file_name=TileConfiguration.txt output=" +inputDirectory2 + " start_x=" + startX + " start_y=" + startY + " start_z=1 start_i=1 channels_for_registration=[Red, Green and Blue] fusion_method=[Linear Blending] fusion_alpha=" + fal + " regression_threshold=" + rth + " max/avg_displacement_threshold=" + mdt + " absolute_displacement_threshold=" + adt + " compute_overlap");			
        File.rename(inputDirectory2 + File.separator + "Stitched Image_1.tif" , inputDirectory2 + File.separator + "StitchedImage.tif");
        open(inputDirectory2 + File.separator + "StitchedImage.tif");
        run("Size...", "width=2620 depth=1 constrain average interpolation=Bilinear");
        wait(100);
        saveAs("Tiff", outputDirectory + File.separator + "StitchedImage.tif");
        close();
        close();
        File.delete(inputDirectory2 + File.separator + "StitchedImage.tif");

        File.copy(inputDirectory2 + File.separator + "TileConfiguration.txt", inputDirectory + File.separator + "TileConfiguration.txt");
        File.copy(inputDirectory2 + File.separator + "TileConfiguration.txt.registered", inputDirectory + File.separator + "TileConfiguration.txt.registered");

        listImg2 = getFileList(inputDirectory2);
        for (j=0; j<=listImg2.length-1; j++){
            x = File.delete(inputDirectory2 + File.separator + listImg2[j]);
        }
        x = File.delete(inputDirectory2);
        wait(1000);
    }
}
//end Stitching loop
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////
}