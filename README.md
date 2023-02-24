# Automated Feature Extraction Tool
The automated feature extraction tool is a powerful project that extracts relevant information such as name, age, house number, guardian name, and registration number from images.

This tool utilizes advanced computer vision techniques, starting by converting PDF pages to images. The images are then passed through two YOLOv5 models. The first model identifies each person's block from a group of people in the image. Once the block is extracted, the second YOLOv5 model extracts the relevant features specified above.

To ensure the accuracy of the data, OCR technology is used to extract the information from the image. The tool is capable of extracting data from multiple images in a fast and efficient manner.

Finally, the extracted data is saved in an Excel format for easy access and further analysis. With its advanced features and functionality, the automated feature extraction tool is a valuable asset for data processing and analysis.
