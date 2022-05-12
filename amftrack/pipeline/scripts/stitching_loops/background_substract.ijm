macro StitchingLoop40even{

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//Asking for the directory where are the folders with the images to be stitched
mainDirectory = "/projects/0/einf914/data/" ;

list = getFileList(mainDirectory);
//Stitching parameters
fileNames       = "Img_r{yy}_c{xx}.tif";

overlap         = "20";  

gridSizeX       = "15"
gridSizeY       = "10"
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
        inputDirectory2  = outputDirectory + File.separator + "Img3";
        File.makeDirectory(inputDirectory2);
        listImg = getFileList(inputDirectory);

        for (j=0; j<=listImg.length-1; j++){
            if(startsWith(listImg[j],"Img_r")){
                print(inputDirectory + File.separator + listImg[j]);
                open(inputDirectory + File.separator + listImg[j]);
                run("Subtract Background...", "rolling=15 light sliding");
                wait(100);
                saveAs("Tiff", inputDirectory2 + File.separator + listImg[j]);
                wait(100);
                close();
            }
        }

        wait(1000);
    }
}
}