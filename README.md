# PyCheck 

**PyCheck** is a tool for 
- STL/STREL monitoring
- Bayesian Statistical Parameter Synthesis for Linear Temporal Properties of Stochastic Models ([TACAS18](https://link.springer.com/chapter/10.1007/978-3-319-89963-3_23))

and other stuff related to the work of 
- Simone Silvetti, Esteco Spa.
- Laura Nenzi, University of Trieste & TU Wien.
- Luca Bortolussi, University of Trieste.

PyCheck ia a partial fork of the project [U-Check](https://link.springer.com/chapter/10.1007/978-3-319-22264-6_6/fulltext.html), plus a conversion from Java to Python.

### How to start 

The reference class for the TACAS paper is BenchMarks.py which is contained in the package pycheck/tesselation/. 
To run it, you need to install python>=3.6 and the following libraries:  
- numpy:1.13
- scipy:0.19
- matplotlib:2.0.2
- numba:0.35
- scikit-learn:0.19

(_the last versions should be ok. In any case we have speciefied the version we used_)


In BenchMarks.py you can find several methods which are reffered to the article. 
Please recall them in the main method at the end of BenchMarks.py (just comment/uncomment them).
