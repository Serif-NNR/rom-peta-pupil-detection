# Real-Time and Accurate Pupil Detection Based Retro-Oriented Mind and Ellipse Trend Analysis

The implementation of the study focusing on detecting the segmentation map of the pupil for wearable and IR eye tracking systems.


<p align="center">
  <img src="https://github.com/user-attachments/assets/5cec9030-b846-4e81-b9c5-20e88ef74e99" width="200"  />
  <img src="https://github.com/user-attachments/assets/900e78df-bbd3-40df-96d7-c97f31f7fdca" width="200"  />
  <img src="https://github.com/user-attachments/assets/203d84da-6dfc-452b-8275-4b62d8d8b253" width="200"  />
</p>


##  Academic side


- If you would like to access the academic article of this study, named **ROM and PETA**, which is derived from the eye-tracking system called **Fixein**:
> [Real-Time and Accurate Pupil Detection Based Retro-Oriented Mind and Ellipse Trend Analysis ](https://www.iieta.org/journals/ts/paper/10.18280/ts.420402 "Real-Time and Accurate Pupil Detection Based Retro-Oriented Mind and Ellipse Trend Analysis ")

<br/>

- For the master's thesis of the comprehensive work called **Fixein**, which also covers ROM-PETA:
> [FIXEIN: A MOBILE AND MODULAR EYE TRACKING SYSTEM](https://tez.yok.gov.tr/UlusalTezMerkezi/TezGoster?key=UjlM15wKZGQW6TLC0pvCty4v6J5vd_fg08_KOX5R3co_5Ju4LjSLjAiZ9sQhj7s5 "FIXEIN: A MOBILE AND MODULAR EYE TRACKING SYSTEM")

<br/><br/>

## Purpose of this Repository

Fixein, as a comprehensive system, encompasses analysis tools, a pupil detector (as shared in this repo), application software, hardware design, hardware prototyping, implementations of other related academic studies, and additional components intended for eye-tracking purposes (e.g., calibration).

The prepared work has been structured in two parts: the pupil detector and the remainder of the system. Through this repository, you can only access the pupil detector implementation and some analysis files.

With the publication of the academic study referring to the entirety of Fixein (i.e., the remainder of the system), it is planned to share all application code in a separate repository and to provide a reference to it here as well.

<br/><br/>

## Usage and Reproduction of the Repository
This repository contains two folders where the analysis and pupil detector codes are shared. The analysis folder includes the scripts used during the development of the analysis processes, while the detector codes were extracted from the main Fixein project. Therefore, if this work is to be reproduced or reused, the folder containing the detector codes should be taken as reference.

Since the detector codes were separated from the main project, the provided code is not directly runnable and requires adaptation. Manual adjustments may include:

- Installing missing libraries that are not bundled,

- Replacing hard-coded file paths with appropriate paths according to the operating system (due to rapid prototyping).

<br/>
If you intend to use only this code file, you may also need additional code files depending on the task. For example:

- Reading camera data and performing normalization if needed,

- Obtaining dataset images in the appropriate manner, format, and resolution,

- Visualizing the input image and the detector outputs, etc.


<br/><br/>

## Notes and Tricks on Code Usage

The main entry point in the code is the file **learning_model_backbone.py**, which contains a class named **xStare_Detector**. Depending on the system type (e.g., monocular tracking), one or two objects of this class may be required.

- Using the *set_traditional_configuration* method of an *xStare_Detector* object, you can provide parameters for traditional methods.

- With the *detect_with_rom method,* you can directly obtain the pupil position from an image.

- If desired, to retrieve the segmentation model's result without applying ROM, the *detect_without_rom method* can be used.

As noted in the academic publication, to automatically generate traditional method parameters, you need to instantiate the **AutoTraditionalConfiguration** class, located in *Auto_Traditional_Configuration/auto_traditional_conf.py*. Using its perform method, it is possible to obtain the traditional method parameters for a given image.


<br/><br/>
## Contact
For any questions regarding the academic publication or the code, please don’t hesitate to contact us at sheriffnnr@gmail.com

We welcome your questions and feedback and are glad to support developments of future studies.

<br/><br/>

