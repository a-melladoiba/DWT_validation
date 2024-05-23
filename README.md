# DWT_validation
 This is the repository that contains all the scripts and files corresponding to the Discrete Wavelet Transform (DWT) method validation.

 Author: Álvaro Mellado Ibáñez (MCs in Computational Biology, BCs in Biochemistry).

 The order of running scripts is the following:
 - Data_preparation: Extracts and modyfies the original data for future analysis.
 - data_to_eurobench: Filters the EMG signal from the steps performed by each patient.
 - spectral_analysis: Performs the DWT to filtered steps samples.
 - data_generator_4_statistics: Generates new populations in order to compare them and stablish a statistical similarity.
 
 Once the populations have been generated, a Student's t-test is performed, in order to stablish statistical similarity.
