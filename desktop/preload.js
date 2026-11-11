const { contextBridge, ipcRenderer } = require("electron");

contextBridge.exposeInMainWorld("picksApi", {
  load: (name) => ipcRenderer.invoke("picks:load", name),
  openSettings: () => ipcRenderer.invoke("settings:open"),
  getSettings: () => ipcRenderer.invoke("settings:get"),
  saveSettings: (s) => ipcRenderer.invoke("settings:save", s),
});
