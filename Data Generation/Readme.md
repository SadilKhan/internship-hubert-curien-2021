# AutoLabelme

Autolabelme takes the json file created in LabelMe and then start template matching for every label.

## To Run
` python3 /path/to/AutoLabelme.py`
## To save the cropped images/vignettes
`python3 /path/to/templatesaver.py <json file>`

## Function for every buttons:
1. Next: Template matching for next label
2. Previous: Template Matching for the previous label
3. Less Boxes: Increases the threshold which results in less number of boxes.
4. More Boxes: Decreases the threshold which results in more number of boxes.
5. Resize Boxes: Increases the threshold which results in resizing the boxes or reduce the number.
6. Finer Resize Boxes: Increases the threshold by little which results in finer resizing the boxes.
7. Save JSON: Saves a JSON file which can be read by LabelMe for further edits.
8. Save Images: Save the cropped vignettes from the image

## Libraries Needed

1. Numpy (`pip install numpy`)
2. PIL (`pip install pillow`)
3. tkinter (`pip install tkinter`)
4. colorsys (`pip install colorsys`)
5. tkmacosx (`pip install tkmacosx`) --> (ONLY FOR MAC USERS)
6. labelme (`pip install labelme`)[WINDOWS/LINUX] (`brew install labelme`)[MAC]

## The Other Py files
1. templatematcher.py --> Matches templates for specified labels or all the labels. Requires JSON file 
