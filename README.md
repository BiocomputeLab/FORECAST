_Code to reproduce some figures in the paper "MPRA: simulation and inference"_ <br />
_The technical information to derive the estimators and simulation algorithm can be found here:_ https://www.overleaf.com/read/zwsdfbbgwtqh

## MPRA- The big data era
The rapid falling costs in both DNA synthesis and sequencing enabled in the last decade the emergence of high-throughput techniques (up to ~10^8 sequences characterized in one experiment) to better explore the phenotype-genotype landscape. Such experimental advances are named Massively Parallel Reporter Assays (MPRAs)
To study genomic regulation, these experiments combine chip synthesized oligos with combinatorial DNA assembly to generate a diversity of genetic constructs whose phenotype is linked to the expression of a fluorescent reporter protein. Indeed, due to the size of these libraries (routinely >10^{4} designs), each member could not be assayed in isolation. However, by pooling the designs and using Fluorescence-Activated Cell Sorting (FACS) followed by DNA sequencing of the separate fluorescent sorted bins, this methodology termed Flow-seq has allowed MPRAs to unravel some design principles for numerous biological systems, including the inference of promoter regulatory logic in eukaryotes, assessing the composability of regulatory elements and the potential contextual effects that can arise and dissecting the influence of RNA secondary structure, codon usage and other factors on translational efficiency in bacteria. 
<img width="1891" alt="flow_seq" src="https://user-images.githubusercontent.com/66125433/95480252-9c257b80-0983-11eb-8d23-5ffe0ee5d128.png">

## Simulation and inference- Aims and context
We built upon the current Flow-seq framework  by focusing on two critical pitfalls often overlooked. First, a practical concern: This complex procedure aggregates multiple highly sensitive experiments: from the size of the library to the number of sequencing reads, any particular choice in experimental parameters during these steps will result in subtle trade-offs whose nature remain unclear. Hence, not only should we identify these trade-offs, but we should also carefully quantify their impact to help inform the design of these experiments.<br />
Secondly, an analytical concern: The scarcity of resources to exploit experimental data has led to suboptimal analysis.  Indeed, reconstructing the signal from the low-resolution binned data is arduous and ill-posed. Intuitively, the binning censors the data as the fluorescence of each cell lies henceforth somewhere within an interval and is not equal to a single value anymore.<br />
The following strategy was adopted to provide some answers to these questions: I first developed a simulation algorithm to generate synthetic data from a typical MPRA, where one can vary the value of several experimental parameters. Two different estimators were then derived, one currently used in most MPRA experiments and an original one  based on Maximum likelihood theory, FLAIR (Flow-seq Likelihood bAsed InfeRence). Analysing the synthetic data across the different simulations with these two estimators allowed us to both assess their performance and draw out insights to inform experimental design.<br />
We then challenged our theoretical approach by confronting our new estimator with a real dataset to both verify the coherence of our claims and highlight the improvement when using FLAIR.  

## Key Results:
![MPRA](https://user-images.githubusercontent.com/66125433/95225452-b9c3db00-07f3-11eb-9dd8-53f57dc7ec1e.jpg)

Fig2 reveals the upward bias of the MOM-based estimators and shows the superiority of a Maximum-likelihood based approach with FLAIR when estimating the parameters of the underlying gamma distribution. <br />
Fig4 can help with preparing the intial library of constructs by setting a lower limit on the number of cells sorted: There's little improvment on the accuracy of the estimates after s=200. 

## Files
__Inference Folder__: <br>
* Inference_lognormal_distribution.py: Script to perform maximum likelihood based inference (FLAIR) on Flow-seq data, applied to Cambray dataset.
* FLAIR_lognormal_Cambray_1-4.ipynb: Jupyter notebook analysing the results of the inference on the 4 merged experimental replicates and elevate the resutls as ground truth to compare with the subsampled experiments.
* Merging_repetitions_Cambray.ipynb :Jupyter notebook showing the filtration steps to clean the flow-seq sequecning raw data, and export it
