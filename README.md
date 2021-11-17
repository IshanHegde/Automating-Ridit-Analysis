# Automating-Ridit-Analysis
This is an example of how to automate Ridit Analysis for a dataset with large amount of questions and many item attributes, with the question response being on likert scale.

For instance, this dataset is about the university faculty perceptions and practices of using Wikipedia as a teaching resource. The university faculties have many interesting attributes such as:

- age 
- gender 
- years of expericence
- their domain of expectrice 

etc... . This survey also has many questions which makes the job of comprehensively analysing the data difficult. This repository is a simple illustration of how to go about automating this tedious task.

# Ridit Analysis

A ridit describes how the distribution of the dependent variable in row i of a contingency table compares relative to an identified distribution. 
Simply, Ridit analysis deals with turining likert scale data into probits/logits.

To convert likert scale into probits/logits, one has to choose a reference group. For survey, it is very unlikely to have a proper reference group. Hence, the whole dataset is taken to be the reference group.

- After a reference data set has been chosen, the reference data set must be converted to a probability function.
- Now, let ![equation](https://latex.codecogs.com/png.latex?x_1%2Cx_2%2Cx_3%2Cx_4%2Cx_5) denote the ordered categories of the preference scale. In our case, the categories are:
- - Strongly disagree, Disagree, Neutral, Agree and Strongly Agree respectively 
- Then, let the probability function p be defined with respect to the reference data set as: 
-  - ![equation](https://latex.codecogs.com/png.latex?p%28X%3Dx_j%29%3DProb%28x_j%29%3D%20%5Ctext%7Bfrequency%20of%20%7D%20x_j)
- Then, the ridit scores, or simply ridits, of the reference data set are then easily calculated as: 
- -![equation](https://latex.codecogs.com/png.latex?w_j%20%3D%200.5p_j%20&plus;%5Csum_%7Bk%3Cj%7D%7Bp_k%7D)
- Now that each of the ordered categories of the preference scale for the reference group have been given ridit scores
- Now, we can calculate the ridit mean, standard error for subclasses partitioned by personal attributes

Note that the mean ridit score for the reference group will always be 0.5

For example, after using the whole dataset as the dataset and calculating the Ridit scores for the preference scale, we partition the dataset by age, compare the statistical properties of hte mean Ridit score of each age group (for example, whether the mean Ridit scores for the age groups deviates from 0.5 significantly)


## Data Source: 

Meseguer-Artola, Antoni & Aibar, Eduard & Lladós-Masllorens, Josep & Minguillón, Julià & Lerga, Maura. (2016). Factors that influence the teaching use of Wikipedia in Higher Education. Journal of the Association for Information Science and Technology. 67. 1224-1232. 10.1002/asi.23488. 

The data used was found at: https://archive.ics.uci.edu/ml/datasets/wiki4he
