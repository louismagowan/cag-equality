<div id="top"></div>
<!--
*** Thanks for checking out the Best-README-Template. If you have a suggestion
*** that would make this better, please fork the repo and create a pull request
*** or simply open an issue with the tag "enhancement".
*** Don't forget to give the project a star!
*** Thanks again! Now go create something AMAZING! :D
-->



<!-- PROJECT SHIELDS -->
<!--
*** I'm using markdown "reference style" links for readability.
*** Reference links are enclosed in brackets [ ] instead of parentheses ( ).
*** See the bottom of this document for the declaration of the reference variables
*** for contributors-url, forks-url, etc. This is an optional, concise syntax you may use.
*** https://www.markdownguide.org/basic-syntax/#reference-style-links
-->
[![Contributors][contributors-shield]][contributors-url]
[![Forks][forks-shield]][forks-url]
[![Stargazers][stars-shield]][stars-url]
[![Issues][issues-shield]][issues-url]
[![MIT License][license-shield]][license-url]
[![LinkedIn][linkedin-shield]][linkedin-url]


<h3 align="center">Centre Assessment Grades in 2020: A Natural Experiment for Investigating Bias in Teacher Judgements</h3>

  <p align="center">
    Code used for capstone project/thesis during MSc in Applied Social Data Science at LSE.
  </p>
</div>





<!-- ABOUT THE PROJECT -->
## About The Project

The COVID-19 pandemic meant that in 2020 students in England were unable to sit their examinations and instead received predicted grades, or “centre assessment grades” (CAGs), from their teachers to allow them to progress. Using the Grading and Admissions Data for England dataset for students in 2020 and 2018-2019, this study treats the use of CAGs as a natural experiment for causally understanding how teacher judgements of academic ability may be biased according to the demographic and socio-economic characteristics of their students. 

A variety of machine learning models (<b>neural networks, support vector regressions, Optuna hyperparameter-optimised LightGBM models</b>) were trained on the 2018-19 data and then used to generate predictions for what the 2020 students were likely to have received had their examinations taken place as usual. The differences between these predictions and the CAGs that students received were calculated and then averaged across students’ different characteristics, revealing what the treatment effects of the use of CAGs for different types of students were likely to have been. 

No evidence of absolute negative bias against students of any demographic or socio-economic characteristic was found, with all groups of students having received higher CAGs than the grades they were likely to have received had they sat their examinations. Some evidence for relative bias was found, with consistent, but insubstantial differences being observed in the treatment effects of certain groups. However, when higher-order interactions of student characteristics were considered, these differences became more substantial. Intersectional perspectives which emphasise the importance of interactions and sub-group differences should be used more widely within quantitative educational equalities research.

Check out the [dashboard of results here](https://datastudio.google.com/reporting/7c49d7ca-ae1c-43cf-a8f7-8e70d969fbad)

## Project Contents

* <b>cag_code.py:<b\> Python file that contains all the code used to analyse data within the Secure Research Service (SRS). The limitations of working in the SRS (due to the sensitivity of the data) meant that code was exported all one file and so is less modular than it could have been. It also meant that any output had to be removed (it was originally in Notebook form).
* <b>results_analysis.Rmd:<b\> RMarkdown file used to produce the conditional average treatment effect graphs for the project.
* <b>prep_for_visualisations.ipynb:<b\> Jupyter Notebook for reformatting and reshaping the CAG data released from the SRS into data that can be more easily uploaded to BigQuery and used in the Data Studio dashboard for the project.

## Built With

* Python: Scikit-Learn, Statsmodels, Scipy, Tensorflow, Keras, LightGBM, Optuna, SHAP, Matplotlib and Seaborn.
* R: Tidyverse, ggplot2 and rlang

<p align="right">(<a href="#top">back to top</a>)</p>





## Prerequisites

In order to work with the same data I did, you will need to apply to become an accredited researcher with the ONS (Office for National Statistics). Then you'll need to submit a project application to work with the [GRADE](https://www.gov.uk/government/publications/grading-and-admissions-data-for-england-grade-framework) dataset. This is a lengthy and involved process, best to allow ~6 months before you'd be able to get started working with the data.

If you do gain access to the dataset, you will also need to request a custom virtual environment to be created for you within the Secure Research Service. Then you'll also need to request whatever non-standard packages you want to be installed there, as you are not able to ingest code or install packages yourself. I requested Keras, Tensorflow, LightGBM, Optuna and Shap to be installed.






<!-- LICENSE -->
## License

Distributed under the MIT License. See `LICENSE.txt` for more information.

<p align="right">(<a href="#top">back to top</a>)</p>



<!-- CONTACT -->
## Contact

Louis Magowan - [Medium Profile](https://medium.com/@louismagowan42)

Project Link: [https://github.com/louismagowan/cag-equality](https://github.com/louismagowan/cag-equality)

<p align="right">(<a href="#top">back to top</a>)</p>



<!-- ACKNOWLEDGMENTS -->
## Acknowledgments

* [Dr Eleanor Knott](https://www.lse.ac.uk/Methodology/People/Academic-Staff/Ellie-Knott/Ellie-Knott)
* [LSE Methodology Department](https://www.lse.ac.uk/methodology)

<p align="right">(<a href="#top">back to top</a>)</p>



<!-- MARKDOWN LINKS & IMAGES -->
<!-- https://www.markdownguide.org/basic-syntax/#reference-style-links -->
[contributors-shield]: https://img.shields.io/github/contributors/louismagowan/cag-equality.svg?style=for-the-badge
[contributors-url]: https://github.com/louismagowan/cag-equality/graphs/contributors
[forks-shield]: https://img.shields.io/github/forks/louismagowan/cag-equality.svg?style=for-the-badge
[forks-url]: https://github.com/louismagowan/cag-equality/network/members
[stars-shield]: https://img.shields.io/github/stars/louismagowan/cag-equality.svg?style=for-the-badge
[stars-url]: https://github.com/louismagowan/cag-equality/stargazers
[issues-shield]: https://img.shields.io/github/issues/louismagowan/cag-equality.svg?style=for-the-badge
[issues-url]: https://github.com/louismagowan/cag-equality/issues
[license-shield]: https://img.shields.io/github/license/louismagowan/cag-equality.svg?style=for-the-badge
[license-url]: https://github.com/louismagowan/cag-equality/blob/master/LICENSE.txt
[linkedin-shield]: https://img.shields.io/badge/-LinkedIn-black.svg?style=for-the-badge&logo=linkedin&colorB=555
[linkedin-url]: https://www.linkedin.com/in/louismagowan/
