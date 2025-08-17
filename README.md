# A Vision-Language Model interpreting Skew-T log-P diagrams

**Flow-chart**  
![Flowchart](Flowchart.png)

**Dataset**:  
Training and validation dataset is at https://osf.io/4n3uh/files/osfstorage

**Demo inference**:   
https://colab.research.google.com/github/hunter3789/VLM-Skew-T/blob/main/source/demo.ipynb
  
**Sample result**:  
"assistant": "The expansive and deep blue region (CAPE) reflects a highly unstable atmosphere, primed for thunderstorm formation.
The blue-shaded area begins low in the atmosphere, indicating that parcels become buoyant with minimal ascent. This favors easy convective initiation.
The CAPE area reaches into the upper levels, suggesting a storm environment with robust vertical development.
The sounding shows potential for cirriform cloud development due to upper-level moisture.
Based on the analysis, the probability of precipitation is low."
![demo](sample.png)
"response (inference)": "A moderately sized blue-shaded region (CAPE) implies moderate instability, which may support thunderstorm development if lifting is present.
The dense yellow layer (CIN) near the surface reflects significant inhibition, likely preventing convection unless strong forcing is present.
The blue-shaded area starts high in the atmosphere, suggesting parcels face strong resistance to ascent and require substantial forcing to become buoyant.
The CAPE area reaches into the upper levels, suggesting a storm environment with robust vertical development.
The wide separation of red and green lines throughout the profile indicates a dry atmosphere with little cloud potential.
Based on the analysis, the probability of precipitation is very high."
