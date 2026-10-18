const { contextBridge, ipcRenderer } = require("electron");

contextBridge.exposeInMainWorld("picksApi", {
  load: () => ipcRenderer.invoke("picks:load"),
  openSettings: () => ipcRenderer.invoke("settings:open"),
  getSettings: () => ipcRenderer.invoke("settings:get"),
  saveSettings: (s) => ipcRenderer.invoke("settings:save", s),
});
