# Data provenance and computational scope

The reference release begins with five frozen files in inputs/. Their exact SHA-256 values are listed in input_manifest.csv.

| File | Role |
| --- | --- |
| species_abundance.csv | 1,090 species by 2,894 communities; composition prediction targets and observed membership |
| species_function_matrix_filtered.csv | 1,090 species by 463 fixed annotation coordinates |
| cmd_sampleMetadata_current.rda | Official curatedMetagenomicData sampleMetadata snapshot downloaded on 27 October 2026 |
| final_full_model.pth | Compatible archived checkpoint used only for dense/grouped numerical and timing checks |
| pathway_perturbation_raw_bc.csv | Historical response archive, retained for provenance; not used for the new attribution claims |

Source metagenomic profiles were assembled through curatedMetagenomicData (Pasolli et al., Nature Methods 2017, DOI 10.1038/nmeth.4468). The metadata snapshot source is https://raw.githubusercontent.com/waldronlab/curatedMetagenomicData/devel/data/sampleMetadata.rda . The original profile resource is https://bioconductor.org/packages/curatedMetagenomicData/ . Resource and original-study terms apply to upstream data and are not replaced by MIT.

The complete historical sequencing filters, reference genome accessions and annotation releases were not recovered. The release supports reproduction from the frozen matrices, rather than reconstruction from raw sequencing. Annotation values describe fixed species potential, not measured activity, expression or flux.

The provenance audit matches 2,777 sample identifiers. The unambiguous study/subject subset has 2,657 samples from 35 studies; 237 unmatched or ambiguous inputs are excluded from the strict holdout. The internal split has 34 known subjects crossing partitions. The independent study holdout uses 544 AsnicarF_2021 samples, with 1,792 training and 321 validation samples from the remaining verified sources. These distinctions are retained in the manuscript.

This folder contains no newly collected participant records or raw human sequencing reads. Input identifiers and metadata are retained to reproduce the published source/split audit. Redistribution and further use remain subject to the applicable source terms.
