### About the Client

* Giga is a global initiative to connect every school to the Internet and every young person to the information, opportunity and choice.

* Some 2.9 billion people in the world do not have access to the Internet. This lack of connectivity means exclusion, marked by the lack of access to the wealth of information available online, fewer resources to learn and to grow, and limited opportunities for the most vulnerable children and youth to fulfill their potential. Closing the digital divide requires global cooperation, leadership, and innovation in finance and technology.

* Giga will ensure every child is equipped with the digital public goods they need, and empowered to shape the future they want. Giga also serves as a platform to create the infrastructure necessary to provide digital connectivity to an entire country, for every community, and for every citizen. It is about using schools to identify demand for connectivity, as well as using schools as an analogy for learning and connecting where the community can come together and support its next generation in a world where we are all increasingly digital, where the skills that are required are not formal ones, necessarily, and where learning happens continuously.

* Giga maintains a real-time map of school connectivity to identify demand for infrastructure and funds, measure progress toward increasing Internet access, and continuously monitor global connectivity. **We have mapped the location of over 1 million schools on our open-source platform: [www.projectconnect.world](http://www.projectconnect.world/)**

* Giga works with governments and advises them **on building affordable and sustainable country-specific models for finance and delivery**, subsidizing market creation costs and incentivizing private sector investment.

* In partnership with industry, and based on the mapping results, Giga will advise on the best possible **technical solutions to provide schools with connectivity**, and **countries with safe, secure, reliable, fit-for-purpose infrastructure to support future digital development needs**.

## Problem Statement

* Develop a Deep Learning base solution to accurately and comprehensively identify school locations in Sudan from high-resolution satellite imagery.

### Data Challenges

* Connecting the unconnected schools.

* School locations records are often inaccurate, incomplete or non-existent

* Traditonal methods are not heavily expensive but some schools are located in remote, inaccessible, and insecurity-prone areas.

* Different file names with same images.

* Duplicate images were present. 

* Schools images were found in non-school images.

### About Data

* Used 224*224 images with zoom level 18(0.6m spatial resolution) for each training sample.

* The imager were collected from MAXAR's imagery archive under NextView license.

* The imagery which were collected from Wordview3 sensor were composited with R, G, B bands using natural composite method.

### Data Labeling

* U and L shape of schools were commonly observed.

* Borders were there in which open field were present.

* Water Tank close to building.

### Super Resolution

* Used Generative Face Prior-Generative Adeversarial Network (GFP-GAN)

* Restores satellite images from low quality to super resolution

* Enhances colours of the school satellite

* Removes the noise from raw satellite images

### Modeling

* Data were 5223(school) & 5223(non-school)

* EfficientDet model perform the with 0.92, 0.81, 0.79 MAP 

* Yolo v5

* First classify images into school and non school using hog features. Then divide the images into 92*92 tiles and using small CNN identfy whether school is present or not. 

*   


