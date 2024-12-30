<h1> CCTV Footage Tracking and Classification</h1> <br>

<h3> Description: </h3> 
<p> This project focusses on taking in any form of video input, can be a live CCTV source from a RTPC url, or a pre-recorded video and classify the people in the screen as either <b>Wearing Coats</b> and <b>Not Wearing Coats</b>. The classification is done in real time and classifies the person in the frame into one of two classes </p>

<h3> Pipeline: </h3><br>
<p> The entire pipeline can be thought of 3 main processes: <br> <h4>1. Object Detection for the human beings in the frame </h4> <h4>2. The tracking of the detected objects through all the frames in the video</h4><h4>3. The classification of the detected objects as either 'wearing a coat' or 'not wearing a coat' </h4></p><br>

<h3> Models Used: </h3><br>
<p> 1. The first 2 processes that involves, tracking of the detected objects has been integrated into one script, thanks to a very useful transformer called <b>Real Time DEtection TRansformer (RTDTER)</b><br> 2. For the classification, I chose a relatively lightweight model that has a good tradeoff between performance and speed, <b>ResNet50</b> on a custom dataset</p>

<h3> About the Dataset: </h3><br>
<p> The making of the dataset involved querying <b>Dalle-e-3</b> with specific prompts for generating images of people with and without coats. Apart from this, a good chunk of the data was <b>scraped</b> from websites like Pexels, ShutterStock and others. </p>
