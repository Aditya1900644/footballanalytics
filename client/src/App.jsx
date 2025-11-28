import React, { useState, useEffect } from 'react';
import axios from 'axios';
import { Play, BarChart2, Map, Activity, UploadCloud } from 'lucide-react';
import { AreaChart, Area, XAxis, YAxis, Tooltip, ResponsiveContainer } from 'recharts';

// API Base URL
const API_URL = "http://localhost:8000";

function App() {
  const [status, setStatus] = useState('idle'); // idle, processing, completed
  const [jobId, setJobId] = useState(null);
  const [progress, setProgress] = useState(0);
  const [data, setData] = useState(null); // Stores video URL, threat data
  const [heatmaps, setHeatmaps] = useState([]);
  const [selectedMap, setSelectedMap] = useState(null);

  // Upload Handler
  const handleUpload = async (e) => {
    const file = e.target.files[0];
    if (!file) return;

    const formData = new FormData();
    formData.append("file", file);
    
    try {
      setStatus('processing');
      const res = await axios.post(`${API_URL}/upload`, formData);
      setJobId(res.data.job_id);
    } catch (err) {
      console.error(err);
      alert("Upload failed");
      setStatus('idle');
    }
  };

  // Polling for Status
  useEffect(() => {
    let interval;
    if (status === 'processing' && jobId) {
      interval = setInterval(async () => {
        const res = await axios.get(`${API_URL}/status/${jobId}`);
        setProgress(res.data.progress);
        
        if (res.data.status === 'completed') {
          setData(res.data.result);
          setStatus('completed');
          
          // Fetch Heatmaps
          const maps = await axios.get(`${API_URL}/heatmaps`);
          setHeatmaps(maps.data);
          if(maps.data.length > 0) setSelectedMap(maps.data[0]);
          
          clearInterval(interval);
        }
      }, 2000);
    }
    return () => clearInterval(interval);
  }, [status, jobId]);

  return (
    <div className="min-h-screen p-8">
      <header className="flex items-center gap-4 mb-8">
        <div className="w-12 h-12 bg-gradient-to-br from-blue-500 to-accent rounded-xl flex items-center justify-center shadow-lg shadow-blue-500/20">
          <Activity className="text-white" />
        </div>
        <div>
          <h1 className="text-3xl font-bold bg-clip-text text-transparent bg-gradient-to-r from-white to-gray-400">
            Football Analytics
          </h1>
          <p className="text-gray-500">Deep Learning Tactical Analysis</p>
        </div>
      </header>

      {/* 1. UPLOAD SCREEN */}
      {status === 'idle' && (
        <div className="flex justify-center mt-20">
          <label className="w-full max-w-xl h-64 border-2 border-dashed border-gray-700 rounded-2xl flex flex-col items-center justify-center cursor-pointer hover:border-accent hover:bg-card/50 transition-all group">
            <UploadCloud className="w-16 h-16 text-gray-500 group-hover:text-accent mb-4" />
            <span className="text-xl font-medium text-gray-300">Click to Upload Match Video</span>
            <span className="text-sm text-gray-600 mt-2">MP4, MOV (Max 500MB)</span>
            <input type="file" className="hidden" onChange={handleUpload} />
          </label>
        </div>
      )}

      {/* 2. PROCESSING SCREEN */}
      {status === 'processing' && (
        <div className="flex flex-col items-center mt-32">
          <div className="w-64 h-4 bg-gray-800 rounded-full overflow-hidden">
            <div 
              className="h-full bg-accent transition-all duration-500" 
              style={{ width: `${progress}%` }}
            />
          </div>
          <h2 className="text-2xl mt-4 font-mono">{progress}% Analyzed</h2>
          <p className="text-gray-500">Running YOLOv11s Inference & Threat Estimation...</p>
        </div>
      )}

      {/* 3. DASHBOARD */}
      {status === 'completed' && data && (
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
          
          {/* Left: Video & Graph */}
          <div className="lg:col-span-2 space-y-6">
            <div className="bg-black rounded-2xl overflow-hidden border border-gray-800 shadow-2xl">
              <video src={data.video_url} controls className="w-full h-auto" />
            </div>
            
            <div className="bg-card p-6 rounded-2xl border border-gray-800">
              <h3 className="text-lg font-semibold mb-4 flex items-center gap-2">
                <Activity size={18} className="text-secondary" /> Match Momentum (xT)
              </h3>
              <div className="h-48 w-full">
                <ResponsiveContainer>
                  <AreaChart data={data.threat_data}>
                    <defs>
                      <linearGradient id="colorThreat" x1="0" y1="0" x2="0" y2="1">
                        <stop offset="5%" stopColor="#ff0055" stopOpacity={0.3}/>
                        <stop offset="95%" stopColor="#ff0055" stopOpacity={0}/>
                      </linearGradient>
                    </defs>
                    <Tooltip 
                      contentStyle={{ backgroundColor: '#111', border: 'none', borderRadius: '8px' }}
                    />
                    <Area type="monotone" dataKey="value" stroke="#ff0055" fill="url(#colorThreat)" />
                  </AreaChart>
                </ResponsiveContainer>
              </div>
            </div>
          </div>

          {/* Right: Stats & Heatmaps */}
          <div className="space-y-6">
            {/* Stats Card */}
            <div className="bg-card p-6 rounded-2xl border border-gray-800">
              <h3 className="text-lg font-semibold mb-4">Quick Stats</h3>
              <div className="grid grid-cols-2 gap-4">
                <div className="bg-dark p-4 rounded-xl">
                  <p className="text-gray-500 text-sm">Pass Predictions</p>
                  <p className="text-2xl font-bold text-white">{data.passes.length}</p>
                </div>
                <div className="bg-dark p-4 rounded-xl">
                  <p className="text-gray-500 text-sm">High Threat Events</p>
                  <p className="text-2xl font-bold text-secondary">
                    {data.threat_data.filter(x => x.value > 50).length}
                  </p>
                </div>
              </div>
            </div>

            {/* Heatmaps */}
            <div className="bg-card p-6 rounded-2xl border border-gray-800">
              <h3 className="text-lg font-semibold mb-4 flex items-center gap-2">
                <Map size={18} className="text-accent" /> Player Heatmaps
              </h3>
              
              {heatmaps.length > 0 ? (
                <>
                  <select 
                    className="w-full bg-dark border border-gray-700 rounded-lg p-3 mb-4 text-white outline-none focus:border-accent"
                    onChange={(e) => {
                        const map = heatmaps.find(h => h.id === e.target.value);
                        setSelectedMap(map);
                    }}
                  >
                    {heatmaps.map(h => (
                      <option key={h.id} value={h.id}>Player {h.id}</option>
                    ))}
                  </select>
                  
                  {selectedMap && (
                    <div className="relative rounded-xl overflow-hidden border border-gray-700">
                      {/* Field Background */}
                      <img 
                        src="https://raw.githubusercontent.com/mradovic38/football_analysis/refs/heads/main/input_videos/field_2d_v2.png" 
                        className="w-full h-auto opacity-50"
                      />
                      {/* Heatmap Overlay */}
                      <img 
                        src={selectedMap.url}
                        className="absolute inset-0 w-full h-full mix-blend-screen"
                      />
                    </div>
                  )}
                </>
              ) : (
                <p className="text-gray-500">No heatmaps generated yet.</p>
              )}
            </div>
          </div>

        </div>
      )}
    </div>
  );
}

export default App;