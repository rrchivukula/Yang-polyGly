args <- commandArgs(trailingOnly=TRUE)

path.samples <- args[1]
path.counts.normalization <- args[2]
path.output <- args[3]

suppressMessages(library(DESeq2))

samples <- read.delim(path.samples, header = T, sep = "\t", check.names = F, comment.char = "#")
samples <- samples[, c("sample", "group")]
samples$group <- factor(samples$group)
# print(samples)

counts.normalization <- read.delim(path.counts.normalization, header = T, sep = "\t", check.names = F, comment.char = "#")
row.names(counts.normalization) <- counts.normalization[, 1]
counts.normalization <- counts.normalization[, -1]
# print(head(counts.normalization))

dds.normalization <- DESeqDataSetFromMatrix(countData = counts.normalization, colData = samples, design = ~ group)
dds.normalization <- estimateSizeFactors(dds.normalization)
print(sizeFactors(dds.normalization))

write.table(as.data.frame(sizeFactors(dds.normalization)), file = path.output, row.names = F, col.names = F, quote = F)
