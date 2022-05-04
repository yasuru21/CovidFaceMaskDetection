# To run our program
- We built and tested this program on windows machines, so we make no claim that it works correctly on a MAC.
Please run this on a Windows machine. 

First, you will need to manually take the mask1_model_resnet101.pth file that is provided
separately in our project zip file and move it to this repository. We were unable to push it To
Github because its a large file. Once you do this, you can run either label_image.py by going to the 
if__name__=='__main__' condition of the code and updating lines 83 and 94 to specify an image path.
(You may need to also manually update line 98 if this file path doesn't lead to a font, but it should on Windows)

You can then view the terminal output with a prediction (with_mask or without_mask) and there should be an annotated 
version of your input image in the repository. To run live video detection, run label_live_cam.py on a Windows machine with 
a webcam installed, and simply position yourself within the frame that appears on your screen. You should notice the frame's border
changes colors according to the model's prediction. 