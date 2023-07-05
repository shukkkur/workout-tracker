<h1 align='center'>Workout Tracking 🏋️</h1>


<p align='center'>
  <a href="https://github.com/shukkkur/workout-tracker/forks"><img src="https://img.shields.io/github/forks/shukkkur/workout-tracker.svg"></a>
  <a href="https://github.com/shukkkur/workout-tracker/stargazers"><img src="https://img.shields.io/github/stars/shukkkur/workout-tracker.svg"></a>
  <a href="https://github.com/shukkkur/workout-tracker/watchers"><img src="https://img.shields.io/github/watchers/shukkkur/workout-tracker.svg"></a>
 
  <br>
  <a href=""><img src="https://img.shields.io/github/last-commit/shukkkur/workout-tracker.svg"></a>
  <img src="https://hits.sh/github.com/shukkkur/workout-tracker.svg"/>

</p>


<p>Count the number of repetitions for 3 exercises: <strong>bench press</strong>, <strong>overhead press</strong> and <strong>dumbbells curl</strong></p>

<h3>How it works</h3>
<ol>
  <li>Using mediapipe process video, detect body landmarks and write it into the csv (<a href="https://github.com/ArtLabss/workout-tracking/blob/main/data_engineering.py">data_engineering.py</a>).</li>
  <li>Train RandomForestClassifier on created csv with four classes: 'bench', 'curl', 'overhead' and 'other'(<a href="https://github.com/ArtLabss/workout-tracking/blob/main/ml_model.py">ml_model.py</a>).</li>
  <li>Process a new video, classify the detected pose using the model (<a href="https://github.com/ArtLabss/workout-tracking/blob/main/BigData.pkl">rf_classifier.pkl</a>) and based on the classification calculate the angle of elbow and shoulder (<a href="https://github.com/ArtLabss/workout-tracking/blob/main/utils.py">calc_angle.py</a>).</li>
  <li>If the calculated angle exceed a certain threshhold, increase counter by 1.
</ol>

<br>

| ![curl_sample](https://github.com/ArtLabss/workout-tracking/blob/c76a33bbaf5ab844852aea4d4806cdd531168792/Input/curl_sample.gif) | ![overhead_sample](https://github.com/ArtLabss/workout-tracking/blob/c76a33bbaf5ab844852aea4d4806cdd531168792/Input/overhead_sample.gif) | ![bench_sample](https://github.com/ArtLabss/workout-tracking/blob/c76a33bbaf5ab844852aea4d4806cdd531168792/Input/bench_sample.gif) |
| :---         |     :---:      |          ---: |


<h3>How to run</h3>

<ol>
  <li>
    Clone this repository
  </li>
  
  ```git
  git clone https://github.com/ArtLabss/workout-tracking.git
  ```
  
  <li>
    Install the requirements using pip 
  </li>
  
  ```python
  pip install -r requirements.txt
  ```
  
  <li>
    Run the following command in the command line
  </li>
  
  ```python
  python prediction.py --input_video_path=Input/bench_test.mp4 --output_video_path=Output/video_output.mp4 --draw_pose=0 --info=0
  ```
  
  <ul>
    <li><code>--input_video_path</code> - path to the input video </li>
    <li><code>--output_video_path</code> - path to the output video, if not specified the video will be created in the same directory as input video</li>
    <li><code>--draw_pose</code> - if set to <code>1</code> the detected poses will be drawn, default is <code>0</code></li>
    <li><code>--info</code> - if <code>1</code> the probabilities of pose classification (<code>[[prob_bench, prob_curl, prob_overhead, prob_other]]</code>), elbow and shoulder angles, as well as which hands is used for calculating the angle will be shown, default is <code>0</code></li>
  </ul>
  
  <br>
  
<p><i>If you stumble upon an <b>error</b> or have any questions feel free to open a new <a href='https://github.com/ArtLabss/workout-tracking/issues'>Issue</a> </i></p>
