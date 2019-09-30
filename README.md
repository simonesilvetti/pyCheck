**PyCheck** is a tool for STL/STREL monitoring and other related things with the works of

-Simone Silvetti, University of Udine and Esteco Spa.

-Laura Nenzi, TU Wien.

-Luca Bortolussi, University of Trieste.

It starts from a partial fork of the project [U-Check](https://link.springer.com/chapter/10.1007/978-3-319-22264-6_6/fulltext.html) and a conversion from Java to Python.
The reason of using Python is only related of its usability. In other words there are lots of package from machine learning, optimisation etc that we have/would/should used/use.

**Istruction for TACAS 2018 Reviewers:** 

The current branch is "master", we will create soon a dedicated brench for the TACAS paper. Untill that moment you can refer to the "master". 
The reference class for the TACAS paper is BenchMarks.py which is contained in the package pycheck/tesselation/. 
To execute it, you need to install python 3.6 and the following python libraries: 
(the last versions should be ok. In any case we have speciefied the version we used)

- numpy:1.13, scipy:0.19, matplotlib:2.0.2, numba:0.35, scikit-learn:0.19

Once you have done it, in BenchMarks.py you can find several methods which are reffered to the article. 
To execute, you have to recall them in the main method at the end of BenchMarks.py (just comment/uncomment them). 
More detailes are specified in BenchMarks.py


