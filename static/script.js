/**
 * CrowdVision AI — Frontend Application Logic
 * Integrates drag-drop upload, Protest Mode calibration, real-time polling HUD,
 * H.264 video playback, and Chart.js interactive timeline analytics.
 */

document.addEventListener('DOMContentLoaded', () => {
    // DOM Elements
    const dropZone = document.getElementById('drop-zone');
    const fileInput = document.getElementById('video-file-input');
    const fileInfoBar = document.getElementById('file-info-bar');
    const selectedFileName = document.getElementById('selected-file-name');
    const selectedFileSize = document.getElementById('selected-file-size');
    const removeFileBtn = document.getElementById('remove-file-btn');
    const uploadForm = document.getElementById('upload-form');
    const startBtn = document.getElementById('start-btn');
    const sampleBtn = document.getElementById('sample-btn');

    // Protest Mode & Controls
    const protestModeToggle = document.getElementById('protest-mode-toggle');
    const protestModeCard = document.getElementById('protest-mode-card');
    const modelSelect = document.getElementById('model-select');
    const confSlider = document.getElementById('conf-slider');
    const confVal = document.getElementById('conf-val');
    const iouSlider = document.getElementById('iou-slider');
    const iouVal = document.getElementById('iou-val');
    const strideSelect = document.getElementById('stride-select');
    const styleSelect = document.getElementById('style-select');
    const imgszSelect = document.getElementById('imgsz-select');
    const enhanceContrastCheck = document.getElementById('enhance-contrast-check');
    const heatmapCheck = document.getElementById('heatmap-check');

    // Sections
    const uploadSection = document.getElementById('upload-section');
    const progressSection = document.getElementById('progress-section');
    const resultsSection = document.getElementById('results-section');

    // Progress Elements
    const progressHeading = document.getElementById('progress-heading');
    const progressBarFill = document.getElementById('progress-bar-fill');
    const statPercent = document.getElementById('stat-percent');
    const statFrames = document.getElementById('stat-frames');
    const statFps = document.getElementById('stat-fps');
    const statLiveCount = document.getElementById('stat-live-count');
    const statUniqueCount = document.getElementById('stat-unique-count');
    const statEta = document.getElementById('stat-eta');
    const cancelBtn = document.getElementById('cancel-btn');

    // Results Elements
    const resVideoName = document.getElementById('res-video-name');
    const resModeBadge = document.getElementById('res-mode-badge');
    const processedVideo = document.getElementById('processed-video');
    const downloadVideoBtn = document.getElementById('download-video-btn');
    const exportJsonBtn = document.getElementById('export-json-btn');
    const newVideoBtn = document.getElementById('new-video-btn');

    const metricMaxCount = document.getElementById('metric-max-count');
    const metricPeakTime = document.getElementById('metric-peak-time');
    const metricUniqueCount = document.getElementById('metric-unique-count');
    const metricAvgCount = document.getElementById('metric-avg-count');
    const metricDuration = document.getElementById('metric-duration');
    const metricSpeedRatio = document.getElementById('metric-speed-ratio');
    const intervalTableBody = document.getElementById('interval-table-body');

    // State Variables
    let selectedFile = null;
    let currentTaskId = null;
    let pollInterval = null;
    let chartInstance = null;
    let latestAnalysisData = null;

    // Protest Mode Preset Sync
    if (protestModeToggle) {
        protestModeToggle.addEventListener('change', (e) => {
            const isProtest = e.target.checked;
            if (isProtest) {
                if (protestModeCard) protestModeCard.classList.remove('inactive');
                if (modelSelect) modelSelect.value = 'yolov8s';
                if (confSlider) { confSlider.value = 0.22; confVal.textContent = '22%'; }
                if (iouSlider) { iouSlider.value = 0.55; iouVal.textContent = '55%'; }
                if (styleSelect) styleSelect.value = 'dots';
                if (imgszSelect) imgszSelect.value = '960';
                if (enhanceContrastCheck) enhanceContrastCheck.checked = true;
            } else {
                if (protestModeCard) protestModeCard.classList.add('inactive');
                if (modelSelect) modelSelect.value = 'yolov8n';
                if (confSlider) { confSlider.value = 0.35; confVal.textContent = '35%'; }
                if (iouSlider) { iouSlider.value = 0.70; iouVal.textContent = '70%'; }
                if (styleSelect) styleSelect.value = 'hybrid';
                if (imgszSelect) imgszSelect.value = '640';
            }
        });
    }

    // Sliders Sync
    if (confSlider && confVal) {
        confSlider.addEventListener('input', (e) => {
            confVal.textContent = `${Math.round(e.target.value * 100)}%`;
        });
    }

    if (iouSlider && iouVal) {
        iouSlider.addEventListener('input', (e) => {
            iouVal.textContent = `${Math.round(e.target.value * 100)}%`;
        });
    }

    // Drag & Drop Handlers
    ['dragenter', 'dragover'].forEach(eventName => {
        dropZone.addEventListener(eventName, (e) => {
            e.preventDefault();
            e.stopPropagation();
            dropZone.classList.add('dragover');
        });
    });

    ['dragleave', 'drop'].forEach(eventName => {
        dropZone.addEventListener(eventName, (e) => {
            e.preventDefault();
            e.stopPropagation();
            dropZone.classList.remove('dragover');
        });
    });

    dropZone.addEventListener('drop', (e) => {
        const files = e.dataTransfer.files;
        if (files && files.length > 0) {
            handleFileSelect(files[0]);
        }
    });

    fileInput.addEventListener('change', (e) => {
        if (e.target.files && e.target.files.length > 0) {
            handleFileSelect(e.target.files[0]);
        }
    });

    removeFileBtn.addEventListener('click', (e) => {
        e.stopPropagation();
        clearFileSelection();
    });

    function formatBytes(bytes) {
        if (bytes === 0) return '0 Bytes';
        const k = 1024;
        const sizes = ['Bytes', 'KB', 'MB', 'GB'];
        const i = Math.floor(Math.log(bytes) / Math.log(k));
        return parseFloat((bytes / Math.pow(k, i)).toFixed(1)) + ' ' + sizes[i];
    }

    function handleFileSelect(file) {
        const validExtensions = ['mp4', 'avi', 'mov', 'mkv', 'webm'];
        const ext = file.name.split('.').pop().toLowerCase();
        if (!validExtensions.includes(ext)) {
            alert(`Unsupported file format (.${ext}). Please select an MP4, AVI, MOV, or MKV video.`);
            return;
        }

        selectedFile = file;
        selectedFileName.textContent = file.name;
        selectedFileSize.textContent = `(${formatBytes(file.size)})`;
        fileInfoBar.classList.remove('hidden');
        startBtn.removeAttribute('disabled');
    }

    function clearFileSelection() {
        selectedFile = null;
        fileInput.value = '';
        fileInfoBar.classList.add('hidden');
        startBtn.setAttribute('disabled', 'true');
    }

    // Form Submission (Upload Video)
    uploadForm.addEventListener('submit', async (e) => {
        e.preventDefault();
        if (!selectedFile) return;

        const formData = new FormData(uploadForm);
        formData.set('video', selectedFile);

        // Ensure toggle states are explicitly attached
        if (protestModeToggle) {
            formData.set('protest_mode', protestModeToggle.checked ? 'true' : 'false');
        }
        if (enhanceContrastCheck) {
            formData.set('enhance_contrast', enhanceContrastCheck.checked ? 'true' : 'false');
        }
        if (heatmapCheck) {
            formData.set('heatmap', heatmapCheck.checked ? 'true' : 'false');
        }

        const isProtest = protestModeToggle ? protestModeToggle.checked : true;
        switchToProgressView(selectedFile.name, isProtest);

        try {
            const response = await fetch('/api/upload', {
                method: 'POST',
                body: formData
            });

            if (!response.ok) {
                const errData = await response.json();
                throw new Error(errData.error || 'Upload failed');
            }

            const data = await response.json();
            currentTaskId = data.task_id;
            startPollingProgress(currentTaskId);
        } catch (error) {
            console.error('Upload Error:', error);
            alert(`Error starting analysis: ${error.message}`);
            resetToUploadView();
        }
    });

    // Try Demo Sample Video
    sampleBtn.addEventListener('click', async () => {
        switchToProgressView('sample_crowd.mp4', true);

        try {
            const response = await fetch('/api/sample', {
                method: 'POST'
            });

            if (!response.ok) {
                const errData = await response.json();
                throw new Error(errData.error || 'Sample creation failed');
            }

            const data = await response.json();
            currentTaskId = data.task_id;
            startPollingProgress(currentTaskId);
        } catch (error) {
            console.error('Sample Error:', error);
            alert(`Error generating sample: ${error.message}`);
            resetToUploadView();
        }
    });

    // Cancel Processing
    cancelBtn.addEventListener('click', async () => {
        if (!currentTaskId) return;
        if (confirm('Are you sure you want to cancel the video analysis?')) {
            try {
                await fetch(`/api/cancel/${currentTaskId}`, { method: 'POST' });
                clearInterval(pollInterval);
                resetToUploadView();
            } catch (err) {
                console.error('Cancel Error:', err);
            }
        }
    });

    // Real-Time Progress Polling
    function startPollingProgress(taskId) {
        if (pollInterval) clearInterval(pollInterval);

        pollInterval = setInterval(async () => {
            try {
                const res = await fetch(`/api/status/${taskId}`);
                if (!res.ok) throw new Error('Status check failed');

                const task = await res.json();

                if (task.status === 'processing' || task.status === 'queued') {
                    const pct = task.progress || 0.0;
                    progressBarFill.style.width = `${pct}%`;
                    statPercent.textContent = `${pct.toFixed(1)}%`;

                    const processed = task.processed_frames || 0;
                    const total = task.total_frames || 0;
                    statFrames.textContent = `${processed} / ${total}`;

                    statFps.textContent = `${task.fps || 0.0} FPS`;
                    statLiveCount.textContent = task.current_count !== undefined ? task.current_count : 0;
                    statUniqueCount.textContent = task.unique_so_far !== undefined ? task.unique_so_far : 0;

                    if (task.time_remaining_est !== undefined && task.time_remaining_est > 0) {
                        statEta.textContent = `${task.time_remaining_est}s remaining`;
                    } else if (pct > 0) {
                        statEta.textContent = 'Finalizing encoding...';
                    } else {
                        statEta.textContent = 'Initializing YOLO neural tracker...';
                    }
                } else if (task.status === 'completed') {
                    clearInterval(pollInterval);
                    progressBarFill.style.width = '100%';
                    statPercent.textContent = '100.0%';
                    setTimeout(() => {
                        renderResults(task);
                    }, 400);
                } else if (task.status === 'failed') {
                    clearInterval(pollInterval);
                    alert(`Analysis failed: ${task.error || 'Unknown error during YOLO processing'}`);
                    resetToUploadView();
                } else if (task.status === 'cancelled') {
                    clearInterval(pollInterval);
                    resetToUploadView();
                }
            } catch (err) {
                console.error('Poll error:', err);
            }
        }, 500);
    }

    // View Transitions
    function switchToProgressView(fileName, isProtest = true) {
        uploadSection.classList.add('hidden');
        resultsSection.classList.add('hidden');
        progressSection.classList.remove('hidden');

        if (progressHeading) {
            progressHeading.textContent = isProtest
                ? 'Analyzing Protest Crowd (Occlusion-Resistant)...'
                : 'Analyzing Crowd Video...';
        }

        // Reset progress indicators
        progressBarFill.style.width = '0%';
        statPercent.textContent = '0.0%';
        statFrames.textContent = '0 / 0';
        statFps.textContent = '0.0 FPS';
        statLiveCount.textContent = '0';
        statUniqueCount.textContent = '0';
        statEta.textContent = 'Initializing...';
    }

    function resetToUploadView() {
        if (pollInterval) clearInterval(pollInterval);
        currentTaskId = null;
        clearFileSelection();

        progressSection.classList.add('hidden');
        resultsSection.classList.add('hidden');
        uploadSection.classList.remove('hidden');
    }

    // Render Results
    function renderResults(data) {
        latestAnalysisData = data;
        progressSection.classList.add('hidden');
        resultsSection.classList.remove('hidden');

        const summary = data.summary || {};
        resVideoName.textContent = data.filename || 'Processed Video';

        if (resModeBadge && summary.mode_used) {
            resModeBadge.textContent = summary.mode_used;
        }

        // 1. Populate Metric Cards
        metricMaxCount.textContent = summary.max_concurrent_count || 0;
        metricPeakTime.textContent = `Peak at ${summary.peak_time_str || '00:00'}`;
        metricUniqueCount.textContent = summary.total_unique_count || 0;
        metricAvgCount.textContent = (summary.avg_count || 0.0).toFixed(1);
        metricDuration.textContent = summary.duration_sec || '0.0';
        metricSpeedRatio.textContent = `Analyzed in ${summary.processing_time_sec || 0}s (${summary.effective_fps || 0} FPS)`;

        // 2. Setup Video Player with cache-buster
        const videoUrl = `${data.video_url}?t=${Date.now()}`;
        processedVideo.src = videoUrl;
        processedVideo.load();

        downloadVideoBtn.href = data.video_url;
        downloadVideoBtn.setAttribute('download', `crowd_tracked_${data.task_id.substring(0, 8)}.mp4`);

        // 3. Render Chart.js Timeline
        renderTimelineChart(data.timeline || []);

        // 4. Render 5-Second Interval Breakdown Table
        renderIntervalTable(data.intervals || []);

        // Scroll smoothly to results
        resultsSection.scrollIntoView({ behavior: 'smooth' });
    }

    function renderTimelineChart(timeline) {
        const ctx = document.getElementById('timeline-chart');
        if (!ctx) return;

        if (chartInstance) {
            chartInstance.destroy();
        }

        const labels = timeline.map(t => t.time_str);
        const peakCounts = timeline.map(t => t.count);
        const avgCounts = timeline.map(t => t.avg_count);

        const gradientCyan = ctx.getContext('2d').createLinearGradient(0, 0, 0, 260);
        gradientCyan.addColorStop(0, 'rgba(6, 182, 212, 0.45)');
        gradientCyan.addColorStop(1, 'rgba(6, 182, 212, 0.02)');

        chartInstance = new Chart(ctx, {
            type: 'line',
            data: {
                labels: labels,
                datasets: [
                    {
                        label: 'Max Crowd Count',
                        data: peakCounts,
                        borderColor: '#06b6d4',
                        backgroundColor: gradientCyan,
                        borderWidth: 2.5,
                        fill: true,
                        tension: 0.35,
                        pointBackgroundColor: '#06b6d4',
                        pointBorderColor: '#0e1422',
                        pointBorderWidth: 2,
                        pointRadius: 3,
                        pointHoverRadius: 6,
                        pointHoverBackgroundColor: '#fff',
                        pointHoverBorderColor: '#06b6d4'
                    },
                    {
                        label: 'Average Crowd',
                        data: avgCounts,
                        borderColor: '#8b5cf6',
                        borderWidth: 1.8,
                        borderDash: [4, 4],
                        fill: false,
                        tension: 0.3,
                        pointRadius: 0,
                        pointHoverRadius: 4
                    }
                ]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                interaction: {
                    mode: 'index',
                    intersect: false
                },
                plugins: {
                    legend: {
                        display: false
                    },
                    tooltip: {
                        backgroundColor: 'rgba(15, 23, 42, 0.95)',
                        titleColor: '#06b6d4',
                        bodyColor: '#f8fafc',
                        borderColor: 'rgba(255, 255, 255, 0.1)',
                        borderWidth: 1,
                        padding: 12,
                        cornerRadius: 8,
                        displayColors: true,
                        callbacks: {
                            title: (items) => `Timestamp: ${items[0].label}`,
                            label: (context) => ` ${context.dataset.label}: ${context.raw} persons`,
                            afterBody: () => 'Click chart to seek video'
                        }
                    }
                },
                scales: {
                    x: {
                        grid: {
                            color: 'rgba(255, 255, 255, 0.04)',
                            drawBorder: false
                        },
                        ticks: {
                            color: '#64748b',
                            font: { family: "'JetBrains Mono', monospace", size: 11 },
                            maxTicksLimit: 12
                        }
                    },
                    y: {
                        beginAtZero: true,
                        suggestedMax: 10,
                        grid: {
                            color: 'rgba(255, 255, 255, 0.06)',
                            drawBorder: false
                        },
                        ticks: {
                            color: '#64748b',
                            font: { family: "'JetBrains Mono', monospace", size: 11 },
                            precision: 0
                        }
                    }
                },
                onClick: (event, elements) => {
                    if (elements && elements.length > 0) {
                        const index = elements[0].index;
                        if (timeline[index]) {
                            const targetSec = timeline[index].second;
                            if (processedVideo) {
                                processedVideo.currentTime = targetSec;
                                processedVideo.play();
                            }
                        }
                    }
                }
            }
        });
    }

    function renderIntervalTable(intervals) {
        if (!intervalTableBody) return;
        intervalTableBody.innerHTML = '';

        if (!intervals || intervals.length === 0) {
            intervalTableBody.innerHTML = `<tr><td colspan="5" style="text-align:center; color:#64748b;">No interval data available</td></tr>`;
            return;
        }

        intervals.forEach(inv => {
            const tr = document.createElement('tr');

            let trendIcon = '➔';
            let trendClass = 'steady';
            if (inv.trend === 'rising') {
                trendIcon = '↗ Rising';
                trendClass = 'rising';
            } else if (inv.trend === 'falling') {
                trendIcon = '↘ Falling';
                trendClass = 'falling';
            } else {
                trendIcon = '➔ Steady';
                trendClass = 'steady';
            }

            tr.innerHTML = `
                <td><strong>${inv.interval_str}</strong></td>
                <td>${inv.min}</td>
                <td>${inv.avg.toFixed(1)}</td>
                <td><span style="color:#06b6d4; font-weight:700;">${inv.max}</span></td>
                <td><span class="trend-badge ${trendClass}">${trendIcon}</span></td>
            `;
            intervalTableBody.appendChild(tr);
        });
    }

    // Export Analysis Data as JSON
    exportJsonBtn.addEventListener('click', () => {
        if (!latestAnalysisData) return;
        const dataStr = 'data:text/json;charset=utf-8,' + encodeURIComponent(JSON.stringify(latestAnalysisData, null, 2));
        const downloadAnchor = document.createElement('a');
        downloadAnchor.setAttribute('href', dataStr);
        downloadAnchor.setAttribute('download', `crowd_analysis_${latestAnalysisData.task_id || 'report'}.json`);
        document.body.appendChild(downloadAnchor);
        downloadAnchor.click();
        downloadAnchor.remove();
    });

    // "Analyze Another Video" Button
    newVideoBtn.addEventListener('click', () => {
        if (processedVideo) {
            processedVideo.pause();
            processedVideo.src = '';
        }
        resetToUploadView();
    });
});
