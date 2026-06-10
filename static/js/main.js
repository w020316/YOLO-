// ===== State =====
let selectedFile = null;
let cameraStream = null;
let currentPage = 'dashboard';
let autoDetectInterval = null;
let isAutoDetecting = false;
let lastResultFilename = '';

// ===== Init =====
document.addEventListener('DOMContentLoaded', () => {
    checkSystemStatus();
    initNavigation();
    initDashboard();
    initDetectPage();
    initCameraPage();
    initVideoPage();
    initBatchPage();
    initHistoryPage();
    initModal();
});

// ===== Toast =====
function showToast(message, type = 'info') {
    const container = document.getElementById('toastContainer');
    const toast = document.createElement('div');
    toast.className = `toast ${type}`;
    toast.textContent = message;
    container.appendChild(toast);
    setTimeout(() => { toast.style.opacity = '0'; toast.style.transform = 'translateX(100%)'; }, 3000);
    setTimeout(() => toast.remove(), 3500);
}

// ===== Modal =====
function initModal() {
    document.getElementById('modalClose').addEventListener('click', () => {
        document.getElementById('modalOverlay').style.display = 'none';
    });
    document.getElementById('modalOverlay').addEventListener('click', e => {
        if (e.target === e.currentTarget) e.currentTarget.style.display = 'none';
    });
}

function showModal(title, bodyHtml) {
    document.getElementById('modalTitle').textContent = title;
    document.getElementById('modalBody').innerHTML = bodyHtml;
    document.getElementById('modalOverlay').style.display = 'flex';
}

// ===== System Status =====
function checkSystemStatus() {
    fetch('/api/status')
        .then(r => r.json())
        .then(data => {
            const el = document.getElementById('systemStatus');
            const dot = el.querySelector('.status-dot');
            if (data.model_loaded) {
                dot.classList.add('online');
                el.querySelector('span').textContent = data.using_custom_model ? '自定义模型已加载' : '预训练模型已加载';
            } else {
                dot.classList.add('offline');
                el.querySelector('span').textContent = '模型未加载';
            }
        })
        .catch(() => {
            const el = document.getElementById('systemStatus');
            el.querySelector('.status-dot').classList.add('offline');
            el.querySelector('span').textContent = '连接失败';
        });
}

// ===== Navigation =====
function initNavigation() {
    const navItems = document.querySelectorAll('.nav-item');
    const pageTitles = {
        dashboard: { title: '数据仪表盘', desc: '系统运行概览与模型训练指标' },
        detect: { title: '图片检测', desc: '上传食品加工场景图片，自动检测人员是否佩戴安全帽' },
        camera: { title: '摄像头检测', desc: '通过摄像头实时捕获画面进行安全帽佩戴检测' },
        video: { title: '视频检测', desc: '上传视频文件，逐帧检测人员安全帽佩戴情况' },
        batch: { title: '批量检测', desc: '同时上传多张图片进行批量检测分析' },
        history: { title: '检测历史', desc: '查看所有历史检测记录和统计信息' }
    };

    navItems.forEach(item => {
        item.addEventListener('click', () => {
            const page = item.dataset.page;
            if (page === currentPage) return;

            navItems.forEach(n => n.classList.remove('active'));
            item.classList.add('active');

            document.querySelectorAll('.page').forEach(p => p.classList.remove('active'));
            document.getElementById(`page-${page}`).classList.add('active');

            document.getElementById('pageTitle').textContent = pageTitles[page].title;
            document.getElementById('pageDesc').textContent = pageTitles[page].desc;

            currentPage = page;

            if (page === 'dashboard') loadDashboard();
            if (page === 'history') loadHistory();
        });
    });
}

// ===== Loading =====
function showLoading(text = '正在处理...') {
    document.getElementById('loadingText').textContent = text;
    document.getElementById('loadingOverlay').style.display = 'flex';
}

function hideLoading() {
    document.getElementById('loadingOverlay').style.display = 'none';
}

// ===== Update Top Stats =====
function updateTopStats(hatCount, nohatCount) {
    document.getElementById('topHatCount').textContent = hatCount;
    document.getElementById('topNohatCount').textContent = nohatCount;
    document.getElementById('topTotalCount').textContent = hatCount + nohatCount;
}

// ===== Render Detection List =====
function renderDetectionList(container, detections) {
    container.innerHTML = '';
    let hatCount = 0, nohatCount = 0;

    detections.forEach(det => {
        if (det.error) {
            container.innerHTML += `<div class="detection-item nohat"><div class="det-icon">!</div><div class="det-info"><div class="det-class">错误</div><div class="det-alert">${det.error}</div></div></div>`;
            return;
        }

        if (det.class === 'hat') hatCount++;
        if (det.class === 'nohat') nohatCount++;

        const isNormal = det.class === 'hat';
        container.innerHTML += `
            <div class="detection-item ${det.class}">
                <div class="det-icon">${isNormal ? 'OK' : '!'}</div>
                <div class="det-info">
                    <div class="det-class">${det.class_cn || det.class} ${isNormal ? '(正常)' : '(异常)'}</div>
                    <div class="det-conf">置信度: ${(det.confidence * 100).toFixed(1)}%</div>
                    ${det.alert ? `<div class="det-alert">${det.alert}</div>` : ''}
                </div>
            </div>`;
    });

    updateTopStats(hatCount, nohatCount);
    return { hatCount, nohatCount };
}

// ===== Dashboard =====
function initDashboard() {
    loadDashboard();
}

function loadDashboard() {
    fetch('/api/dashboard')
        .then(r => r.json())
        .then(data => {
            if (!data.success) return;

            const ov = data.overview;
            document.getElementById('dashTotalDet').textContent = ov.total_detections;
            document.getElementById('dashTotalHat').textContent = ov.total_hat;
            document.getElementById('dashTotalNohat').textContent = ov.total_nohat;
            document.getElementById('dashAlertRate').textContent = ov.alert_rate + '%';

            updateTopStats(ov.total_hat, ov.total_nohat);

            // Model info
            fetch('/api/status').then(r => r.json()).then(status => {
                document.getElementById('miStatus').textContent = status.using_custom_model ? '自定义模型' : '预训练模型';
                document.getElementById('miDevice').textContent = status.device === 'cuda' ? 'GPU' : 'CPU';
                document.getElementById('miSize').textContent = (status.model_size_mb || 0) + ' MB';
                document.getElementById('miTrain').textContent = (status.dataset?.train_images || 0) + ' 张';
                document.getElementById('miVal').textContent = (status.dataset?.val_images || 0) + ' 张';
            });

            // Training metrics
            const tm = data.training_metrics;
            if (tm.final_metrics) {
                document.getElementById('miMap50').textContent = (tm.final_metrics.mAP50 * 100).toFixed(1) + '%';
                document.getElementById('miMap95').textContent = (tm.final_metrics.mAP50_95 * 100).toFixed(1) + '%';
                document.getElementById('miPrec').textContent = (tm.final_metrics.precision * 100).toFixed(1) + '%';
                document.getElementById('miRecall').textContent = (tm.final_metrics.recall * 100).toFixed(1) + '%';
                document.getElementById('miEpochs').textContent = tm.total_epochs || '-';
            }

            // Training plots
            const plotBase = '/api/training_plots/';
            document.getElementById('plotResults').src = plotBase + 'results.png';
            document.getElementById('plotPR').src = plotBase + 'BoxPR_curve.png';
            document.getElementById('plotF1').src = plotBase + 'BoxF1_curve.png';
            document.getElementById('plotPred').src = plotBase + 'val_batch0_pred.jpg';

            // Type distribution
            const ts = data.type_stats;
            const typeDist = document.getElementById('typeDistribution');
            const typeLabels = { image: '图片检测', camera: '摄像头', video: '视频检测', batch: '批量检测' };
            const typeColors = { image: '#00D4AA', camera: '#00A3FF', video: '#FFA502', batch: '#8B5CF6' };
            const total = Object.values(ts).reduce((a, b) => a + b, 0) || 1;

            typeDist.innerHTML = Object.entries(ts).map(([key, val]) => `
                <div class="type-bar-item">
                    <div class="type-bar-value">${val}</div>
                    <div class="type-bar" style="background:${typeColors[key]};width:${Math.max(val / total * 100, 2)}%"></div>
                    <div class="type-bar-label">${typeLabels[key]}</div>
                </div>
            `).join('');
        })
        .catch(err => console.error('Dashboard load failed:', err));
}

// ===== Detect Page =====
function initDetectPage() {
    const uploadZone = document.getElementById('uploadZone');
    const fileInput = document.getElementById('fileInput');
    const previewContainer = document.getElementById('previewContainer');
    const previewImage = document.getElementById('previewImage');
    const detectBtn = document.getElementById('detectBtn');
    const clearPreviewBtn = document.getElementById('clearPreviewBtn');
    const confSlider = document.getElementById('confSlider');
    const confValue = document.getElementById('confValue');

    uploadZone.addEventListener('click', () => fileInput.click());

    ['dragover', 'dragleave', 'drop'].forEach(evt => {
        uploadZone.addEventListener(evt, e => {
            e.preventDefault();
            if (evt === 'dragover') uploadZone.classList.add('dragover');
            else uploadZone.classList.remove('dragover');
            if (evt === 'drop' && e.dataTransfer.files.length > 0) handleFile(e.dataTransfer.files[0]);
        });
    });

    fileInput.addEventListener('change', e => {
        if (e.target.files.length > 0) handleFile(e.target.files[0]);
    });

    function handleFile(file) {
        if (!file.type.startsWith('image/')) { showToast('请上传图片文件', 'warning'); return; }
        selectedFile = file;
        const reader = new FileReader();
        reader.onload = e => {
            previewImage.src = e.target.result;
            previewContainer.style.display = 'block';
            uploadZone.style.display = 'none';
        };
        reader.readAsDataURL(file);
    }

    confSlider.addEventListener('input', e => { confValue.textContent = e.target.value; });

    detectBtn.addEventListener('click', () => {
        if (!selectedFile) return;
        showLoading('正在检测...');

        const formData = new FormData();
        formData.append('image', selectedFile);
        formData.append('conf', confSlider.value);

        fetch('/api/detect', { method: 'POST', body: formData })
            .then(r => r.json())
            .then(data => {
                hideLoading();
                if (data.success) {
                    lastResultFilename = data.filename;
                    document.getElementById('resultContent').style.display = 'block';
                    document.querySelector('.result-empty').style.display = 'none';
                    document.getElementById('resultImage').src = data.result_image;
                    document.getElementById('inferenceTime').textContent = `推理耗时: ${data.inference_time}ms`;
                    const stats = renderDetectionList(document.getElementById('detectionList'), data.detections);

                    const alertBanner = document.getElementById('alertBanner');
                    if (stats.nohatCount > 0) {
                        alertBanner.style.display = 'flex';
                        document.getElementById('alertText').textContent = `检测到 ${stats.nohatCount} 名未佩戴帽子的工作人员！`;
                        showToast(`异常警告: 检测到 ${stats.nohatCount} 名未佩戴帽子人员`, 'error');
                    } else {
                        alertBanner.style.display = 'none';
                        showToast('检测完成，所有人员均佩戴帽子', 'success');
                    }
                } else {
                    showToast('检测失败: ' + (data.error || '未知错误'), 'error');
                }
            })
            .catch(err => { hideLoading(); showToast('请求失败: ' + err.message, 'error'); });
    });

    document.getElementById('downloadResultBtn').addEventListener('click', () => {
        if (lastResultFilename) window.open('/api/download_result/' + lastResultFilename, '_blank');
    });

    clearPreviewBtn.addEventListener('click', () => {
        selectedFile = null;
        fileInput.value = '';
        lastResultFilename = '';
        previewContainer.style.display = 'none';
        uploadZone.style.display = 'block';
        document.getElementById('resultContent').style.display = 'none';
        document.querySelector('.result-empty').style.display = 'flex';
        document.getElementById('alertBanner').style.display = 'none';
        updateTopStats(0, 0);
    });
}

// ===== Camera Page =====
function initCameraPage() {
    const openCameraBtn = document.getElementById('openCameraBtn');
    const captureBtn = document.getElementById('captureBtn');
    const autoDetectBtn = document.getElementById('autoDetectBtn');
    const closeCameraBtn = document.getElementById('closeCameraBtn');
    const video = document.getElementById('cameraVideo');
    const canvas = document.getElementById('cameraCanvas');
    const overlay = document.getElementById('cameraOverlay');
    const liveBadge = document.getElementById('cameraLiveBadge');
    const cameraConfSlider = document.getElementById('cameraConfSlider');
    const cameraConfValue = document.getElementById('cameraConfValue');
    const autoIntervalSlider = document.getElementById('autoIntervalSlider');
    const autoIntervalValue = document.getElementById('autoIntervalValue');

    cameraConfSlider.addEventListener('input', e => { cameraConfValue.textContent = e.target.value; });
    autoIntervalSlider.addEventListener('input', e => { autoIntervalValue.textContent = e.target.value; });

    openCameraBtn.addEventListener('click', async () => {
        try {
            cameraStream = await navigator.mediaDevices.getUserMedia({ video: { width: 1280, height: 720 } });
            video.srcObject = cameraStream;
            overlay.style.display = 'none';
            liveBadge.style.display = 'block';
            captureBtn.disabled = false;
            autoDetectBtn.disabled = false;
            closeCameraBtn.disabled = false;
            showToast('摄像头已打开', 'success');
        } catch (err) {
            showToast('无法访问摄像头: ' + err.message, 'error');
        }
    });

    captureBtn.addEventListener('click', () => {
        canvas.width = video.videoWidth;
        canvas.height = video.videoHeight;
        canvas.getContext('2d').drawImage(video, 0, 0);
        canvas.toBlob(blob => {
            if (!blob) return;
            showLoading('正在检测...');
            const formData = new FormData();
            formData.append('image', blob, 'camera_capture.jpg');
            formData.append('conf', cameraConfSlider.value);

            fetch('/api/detect', { method: 'POST', body: formData })
                .then(r => r.json())
                .then(data => {
                    hideLoading();
                    if (data.success) {
                        lastResultFilename = data.filename;
                        const resultDiv = document.getElementById('cameraResult');
                        resultDiv.style.display = 'block';
                        document.getElementById('cameraResultImage').src = data.result_image;
                        document.getElementById('cameraInferenceTime').textContent = `推理耗时: ${data.inference_time}ms`;
                        const stats = renderDetectionList(document.getElementById('cameraDetectionList'), data.detections);
                        if (stats.nohatCount > 0) showToast(`异常: ${stats.nohatCount}名未佩戴帽子`, 'error');
                        else showToast('检测正常', 'success');
                    }
                })
                .catch(err => { hideLoading(); showToast('请求失败', 'error'); });
        }, 'image/jpeg', 0.95);
    });

    autoDetectBtn.addEventListener('click', () => {
        if (isAutoDetecting) {
            clearInterval(autoDetectInterval);
            autoDetectInterval = null;
            isAutoDetecting = false;
            autoDetectBtn.textContent = '自动检测';
            autoDetectBtn.classList.remove('btn-outline');
            autoDetectBtn.classList.add('btn-danger');
            showToast('自动检测已停止', 'info');
        } else {
            isAutoDetecting = true;
            autoDetectBtn.textContent = '停止自动';
            autoDetectBtn.classList.remove('btn-danger');
            autoDetectBtn.classList.add('btn-outline');
            showToast('自动检测已启动', 'success');

            const doAutoDetect = () => {
                if (!cameraStream || !isAutoDetecting) return;
                canvas.width = video.videoWidth;
                canvas.height = video.videoHeight;
                canvas.getContext('2d').drawImage(video, 0, 0);
                canvas.toBlob(blob => {
                    if (!blob) return;
                    const formData = new FormData();
                    formData.append('image', blob, 'auto_capture.jpg');
                    formData.append('conf', cameraConfSlider.value);
                    fetch('/api/detect', { method: 'POST', body: formData })
                        .then(r => r.json())
                        .then(data => {
                            if (data.success) {
                                lastResultFilename = data.filename;
                                document.getElementById('cameraResult').style.display = 'block';
                                document.getElementById('cameraResultImage').src = data.result_image;
                                document.getElementById('cameraInferenceTime').textContent = `推理耗时: ${data.inference_time}ms`;
                                const stats = renderDetectionList(document.getElementById('cameraDetectionList'), data.detections);
                                if (stats.nohatCount > 0) showToast(`自动检测异常: ${stats.nohatCount}名未佩戴帽子`, 'error');
                            }
                        }).catch(() => {});
                }, 'image/jpeg', 0.95);
            };

            doAutoDetect();
            autoDetectInterval = setInterval(doAutoDetect, parseInt(autoIntervalSlider.value) * 1000);
        }
    });

    closeCameraBtn.addEventListener('click', () => {
        if (isAutoDetecting) {
            clearInterval(autoDetectInterval);
            isAutoDetecting = false;
            autoDetectBtn.textContent = '自动检测';
            autoDetectBtn.classList.remove('btn-outline');
            autoDetectBtn.classList.add('btn-danger');
        }
        if (cameraStream) {
            cameraStream.getTracks().forEach(t => t.stop());
            cameraStream = null;
        }
        video.srcObject = null;
        overlay.style.display = 'flex';
        liveBadge.style.display = 'none';
        captureBtn.disabled = true;
        autoDetectBtn.disabled = true;
        closeCameraBtn.disabled = true;
        showToast('摄像头已关闭', 'info');
    });

    document.getElementById('downloadCameraBtn').addEventListener('click', () => {
        if (lastResultFilename) window.open('/api/download_result/' + lastResultFilename, '_blank');
    });
}

// ===== Video Page =====
function initVideoPage() {
    const zone = document.getElementById('videoUploadZone');
    const input = document.getElementById('videoInput');

    zone.addEventListener('click', () => input.click());
    zone.addEventListener('dragover', e => { e.preventDefault(); zone.classList.add('dragover'); });
    zone.addEventListener('dragleave', () => zone.classList.remove('dragover'));
    zone.addEventListener('drop', e => {
        e.preventDefault(); zone.classList.remove('dragover');
        if (e.dataTransfer.files.length > 0) processVideo(e.dataTransfer.files[0]);
    });
    input.addEventListener('change', e => {
        if (e.target.files.length > 0) processVideo(e.target.files[0]);
    });

    function processVideo(file) {
        showLoading('正在处理视频，请耐心等待...');
        document.getElementById('videoProgress').style.display = 'block';
        zone.style.display = 'none';

        const formData = new FormData();
        formData.append('video', file);
        formData.append('conf', '0.25');

        fetch('/api/video_detect', { method: 'POST', body: formData })
            .then(r => r.json())
            .then(data => {
                hideLoading();
                document.getElementById('videoProgress').style.display = 'none';

                if (data.success) {
                    document.getElementById('videoResult').style.display = 'block';
                    document.getElementById('videoFrames').textContent = data.total_frames;
                    document.getElementById('videoHatCount').textContent = data.hat_total;
                    document.getElementById('videoNohatCount').textContent = data.nohat_total;
                    document.getElementById('videoTime').textContent = (data.processing_time / 1000).toFixed(1) + 's';
                    document.getElementById('resultVideo').src = data.result_video;

                    updateTopStats(data.hat_total, data.nohat_total);

                    const samplesDiv = document.getElementById('videoSamples');
                    samplesDiv.innerHTML = '';
                    if (data.sample_frames) {
                        data.sample_frames.forEach(frame => {
                            samplesDiv.innerHTML += `<img src="data:image/jpeg;base64,${frame}" alt="sample">`;
                        });
                    }

                    if (data.has_alert) showToast(`视频检测完成，发现 ${data.nohat_total} 处异常`, 'warning');
                    else showToast('视频检测完成，全部正常', 'success');
                } else {
                    showToast('视频处理失败: ' + (data.error || '未知错误'), 'error');
                    zone.style.display = 'block';
                }
            })
            .catch(err => {
                hideLoading();
                document.getElementById('videoProgress').style.display = 'none';
                zone.style.display = 'block';
                showToast('请求失败: ' + err.message, 'error');
            });
    }
}

// ===== Batch Page =====
function initBatchPage() {
    const zone = document.getElementById('batchUploadZone');
    const input = document.getElementById('batchInput');
    const batchConfSlider = document.getElementById('batchConfSlider');
    const batchConfValue = document.getElementById('batchConfValue');

    batchConfSlider.addEventListener('input', e => { batchConfValue.textContent = e.target.value; });

    zone.addEventListener('click', () => input.click());
    zone.addEventListener('dragover', e => { e.preventDefault(); zone.classList.add('dragover'); });
    zone.addEventListener('dragleave', () => zone.classList.remove('dragover'));
    zone.addEventListener('drop', e => {
        e.preventDefault(); zone.classList.remove('dragover');
        if (e.dataTransfer.files.length > 0) processBatch(e.dataTransfer.files);
    });
    input.addEventListener('change', e => {
        if (e.target.files.length > 0) processBatch(e.target.files);
    });

    function processBatch(files) {
        showLoading('正在批量检测...');
        document.getElementById('batchProgress').style.display = 'block';
        zone.style.display = 'none';

        const formData = new FormData();
        for (let i = 0; i < files.length; i++) formData.append('images', files[i]);
        formData.append('conf', batchConfSlider.value);

        fetch('/api/batch_detect', { method: 'POST', body: formData })
            .then(r => r.json())
            .then(data => {
                hideLoading();
                document.getElementById('batchProgress').style.display = 'none';

                if (data.success) {
                    const resultsDiv = document.getElementById('batchResults');
                    resultsDiv.innerHTML = '';
                    let totalHat = 0, totalNohat = 0;

                    data.results.forEach(item => {
                        totalHat += item.hat_count;
                        totalNohat += item.nohat_count;
                        resultsDiv.innerHTML += `
                            <div class="batch-item" onclick="showModal('检测结果 - ${item.filename}', '<img src=\\'${item.result_image}\\' alt=\\'result\\'><div style=\\'margin-top:12px\\'>正常: ${item.hat_count} | 异常: ${item.nohat_count} | 耗时: ${item.inference_time}ms</div>')">
                                <img src="${item.result_image}" alt="result">
                                <div class="batch-item-info">
                                    <div class="batch-item-name">${item.filename}</div>
                                    <div class="batch-item-stats">
                                        <span class="hat-stat">正常: ${item.hat_count}</span>
                                        <span class="nohat-stat">异常: ${item.nohat_count}</span>
                                        <span style="color:var(--text-muted)">${item.inference_time}ms</span>
                                    </div>
                                </div>
                            </div>`;
                    });

                    document.getElementById('batchSummary').style.display = 'grid';
                    document.getElementById('batchTotal').textContent = data.total;
                    document.getElementById('batchHatTotal').textContent = totalHat;
                    document.getElementById('batchNohatTotal').textContent = totalNohat;
                    document.getElementById('batchTotalTime').textContent = data.total_time + 'ms';
                    updateTopStats(totalHat, totalNohat);

                    if (totalNohat > 0) showToast(`批量检测完成，${totalNohat}处异常`, 'warning');
                    else showToast('批量检测完成，全部正常', 'success');
                } else {
                    showToast('批量检测失败: ' + (data.error || '未知错误'), 'error');
                    zone.style.display = 'block';
                }
            })
            .catch(err => {
                hideLoading();
                document.getElementById('batchProgress').style.display = 'none';
                zone.style.display = 'block';
                showToast('请求失败: ' + err.message, 'error');
            });
    }
}

// ===== History Page =====
function initHistoryPage() {
    document.getElementById('clearHistoryBtn').addEventListener('click', () => {
        if (confirm('确定要清空所有检测历史吗？')) {
            fetch('/api/history/clear', { method: 'POST' })
                .then(() => { loadHistory(); showToast('历史已清空', 'info'); })
                .catch(err => showToast('清空失败', 'error'));
        }
    });

    document.getElementById('exportReportBtn').addEventListener('click', () => {
        window.open('/api/export_report', '_blank');
        showToast('正在导出报告...', 'info');
    });

    document.getElementById('historyFilter').addEventListener('change', () => loadHistory());
    document.getElementById('historyAlertFilter').addEventListener('change', () => loadHistory());
}

function loadHistory() {
    const filterType = document.getElementById('historyFilter').value;
    const filterAlert = document.getElementById('historyAlertFilter').value;

    let url = '/api/history?limit=50';
    if (filterType) url += '&type=' + filterType;
    if (filterAlert) url += '&alert=' + filterAlert;

    fetch(url)
        .then(r => r.json())
        .then(data => {
            const list = document.getElementById('historyList');
            if (!data.records || data.records.length === 0) {
                list.innerHTML = '<div class="history-empty"><svg width="48" height="48" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1"><circle cx="12" cy="12" r="10"/><polyline points="12 6 12 12 16 14"/></svg><p>暂无检测记录</p></div>';
                return;
            }

            const typeLabels = { image: '图片', camera: '摄像头', video: '视频', batch: '批量' };
            list.innerHTML = '';
            data.records.forEach(record => {
                const badge = record.has_alert
                    ? '<span class="history-badge alert">异常</span>'
                    : '<span class="history-badge normal">正常</span>';

                const typeLabel = typeLabels[record.detection_type] || '图片';

                list.innerHTML += `
                    <div class="history-item" onclick="showHistoryDetail('${record.id}')">
                        <img class="history-thumb" src="${record.result_path}" alt="thumb" onerror="this.style.display='none'">
                        <div class="history-info">
                            <div class="history-time">${record.timestamp} · ${typeLabel}</div>
                            <div class="history-filename">${record.filename}</div>
                            <div class="history-stats">
                                <span class="hat-count">正常: ${record.hat_count}</span>
                                <span class="nohat-count">异常: ${record.nohat_count}</span>
                            </div>
                        </div>
                        ${badge}
                        <button class="history-delete-btn" onclick="event.stopPropagation(); deleteRecord('${record.id}')" title="删除">&times;</button>
                    </div>`;
            });
        })
        .catch(err => console.error('Load history failed:', err));
}

function deleteRecord(id) {
    fetch(`/api/history/${id}`, { method: 'DELETE' })
        .then(() => { loadHistory(); showToast('记录已删除', 'info'); })
        .catch(err => showToast('删除失败', 'error'));
}

function showHistoryDetail(id) {
    fetch('/api/history?limit=200')
        .then(r => r.json())
        .then(data => {
            const record = data.records.find(r => r.id === id);
            if (!record) return;

            let detHtml = '';
            if (record.detections) {
                record.detections.forEach(d => {
                    if (d.error) return;
                    const cls = d.class === 'hat' ? '正常' : '异常';
                    const color = d.class === 'hat' ? 'var(--success)' : 'var(--danger)';
                    detHtml += `<div style="padding:8px;border-left:3px solid ${color};margin:4px 0;background:var(--bg-secondary);border-radius:4px">${d.class_cn} · ${(d.confidence * 100).toFixed(1)}%</div>`;
                });
            }

            showModal('检测详情 - ' + record.filename, `
                <img src="${record.result_path}" alt="result" style="width:100%;border-radius:8px;margin-bottom:16px">
                <div style="display:grid;grid-template-columns:1fr 1fr;gap:12px;margin-bottom:16px">
                    <div style="padding:12px;background:var(--bg-secondary);border-radius:8px;text-align:center">
                        <div style="font-size:24px;font-weight:700;color:var(--success);font-family:Orbitron">${record.hat_count}</div>
                        <div style="font-size:12px;color:var(--text-muted)">正常</div>
                    </div>
                    <div style="padding:12px;background:var(--bg-secondary);border-radius:8px;text-align:center">
                        <div style="font-size:24px;font-weight:700;color:var(--danger);font-family:Orbitron">${record.nohat_count}</div>
                        <div style="font-size:12px;color:var(--text-muted)">异常</div>
                    </div>
                </div>
                <div style="margin-bottom:8px;font-weight:600">检测结果列表</div>
                ${detHtml || '<p style="color:var(--text-muted)">无详细检测数据</p>'}
            `);
        });
}
