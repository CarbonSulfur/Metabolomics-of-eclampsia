# 安装必要的包（如果尚未安装）
if (!require("MALDIquant")) install.packages("MALDIquant")
if (!require("MALDIquantForeign")) install.packages("MALDIquantForeign")

# 加载包
library(MALDIquant)
library(MALDIquantForeign)

# 设置内存管理
gc()  # 强制垃圾回收
memory.limit(size=8000)  # 增加R可用内存（Windows系统）

# 设置文件夹路径
file_path <- "D:\\Desktop\\科技实习与创新\\R语言\\txt"
print(0)

# 分批读取和处理文件
batch_size <- 20  # 每批处理的文件数
file_list <- list.files(file_path, pattern=".txt", full.names=TRUE)
n_batches <- ceiling(length(file_list) / batch_size)

# 创建空矩阵存储最终结果
first_spectrum <- importTxt(file_list[1])[[1]]
n_mass_points <- length(first_spectrum@mass)
n_total_files <- length(file_list)

# 预分配结果矩阵
smoothed_matrix <- matrix(0, nrow=n_total_files, ncol=n_mass_points)
baseline_matrix <- matrix(0, nrow=n_total_files, ncol=n_mass_points)
colnames(smoothed_matrix) <- first_spectrum@mass
colnames(baseline_matrix) <- first_spectrum@mass

# 函数：处理负值强度
handleNegativeIntensities <- function(spectrum) {
    spectrum@intensity[spectrum@intensity < 0] <- 0
    return(spectrum)
}

# 分批处理
for(batch in 1:n_batches) {
    start_idx <- (batch-1) * batch_size + 1
    end_idx <- min(batch * batch_size, length(file_list))
    current_files <- file_list[start_idx:end_idx]
    
    print(paste("Processing batch", batch, "of", n_batches))
    
    # 读取当前批次的文件
    spectra <- importTxt(current_files, verbose=FALSE)
    
    # 平滑滤波
    spectra_smoothed <- smoothIntensity(spectra, method="SavitzkyGolay", halfWindowSize=10)
    spectra_smoothed <- lapply(spectra_smoothed, handleNegativeIntensities)
    
    # 基线矫正
    suppressWarnings({
        spectra_baseline <- removeBaseline(spectra_smoothed, method="SNIP", iterations=1000)
        spectra_baseline <- lapply(spectra_baseline, handleNegativeIntensities)
    })
    
    # 存储结果
    for(i in 1:length(spectra_smoothed)) {
        smoothed_matrix[start_idx+i-1,] <- spectra_smoothed[[i]]@intensity
        baseline_matrix[start_idx+i-1,] <- spectra_baseline[[i]]@intensity
    }
    
    # 清理内存
    rm(spectra, spectra_smoothed, spectra_baseline)
    gc()
}

# # 保存处理后的数据
# write.csv(t(smoothed_matrix), file="smoothed_data.csv")
# write.csv(t(baseline_matrix), file="baseline_data.csv")

# 分批进行峰检测
peaks_list <- list()
for(batch in 1:n_batches) {
    start_idx <- (batch-1) * batch_size + 1
    end_idx <- min(batch * batch_size, length(file_list))
    
    print(paste("Detecting peaks for batch", batch, "of", n_batches))
    
    # 创建临时质谱对象
    temp_spectra <- vector("list", length=end_idx-start_idx+1)
    for(i in 1:(end_idx-start_idx+1)) {
        temp_spectra[[i]] <- createMassSpectrum(
            mass = as.numeric(colnames(baseline_matrix)),
            intensity = baseline_matrix[start_idx+i-1,]
        )
    }
    
    # 峰检测
    batch_peaks <- detectPeaks(temp_spectra, method="MAD", SNR=3, halfWindowSize=50)
    peaks_list <- c(peaks_list, batch_peaks)
    
    # 清理内存
    rm(temp_spectra, batch_peaks)
    gc()
}

# 峰对齐
peaks_aligned <- binPeaks(peaks_list, tolerance=0.002)

# 生成峰值矩阵
peak_matrix <- intensityMatrix(peaks_aligned)
peak_matrix[is.na(peak_matrix)] <- 0

# 保存结果
write.csv(t(peak_matrix), file="peak_matrix.csv")

# 输出峰值个数
peak_counts <- sapply(peaks_aligned, length)
print(peak_counts)