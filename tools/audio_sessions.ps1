# Lists every active playback session per output device (like the Windows
# Volume Mixer) with its process, mute state, volume and current peak level.
#   powershell -ExecutionPolicy Bypass -File tools\audio_sessions.ps1 [-Seconds 10] [-Unmute chrome,msedge]
# -Seconds  keep sampling and report the highest peak (play something meanwhile)
# -Unmute   unmute the sessions of these process names; waits up to -Seconds
#           for one to appear, since browsers only hold a session while playing
param([double]$Seconds = 0, [string[]]$Unmute = @())

Add-Type -Language CSharp @"
using System;
using System.Collections.Generic;
using System.Runtime.InteropServices;

[Guid("A95664D2-9614-4F35-A746-DE8DB63617E6"), InterfaceType(ComInterfaceType.InterfaceIsIUnknown)]
interface IMMDeviceEnumerator {
    int EnumAudioEndpoints(int dataFlow, int stateMask, out IMMDeviceCollection devices);
    int GetDefaultAudioEndpoint(int dataFlow, int role, out IMMDevice device);
}
[Guid("0BD7A1BE-7A1A-44DB-8397-CC5392387B5E"), InterfaceType(ComInterfaceType.InterfaceIsIUnknown)]
interface IMMDeviceCollection {
    int GetCount(out int count);
    int Item(int index, out IMMDevice device);
}
[Guid("D666063F-1587-4E43-81F1-B948E807363F"), InterfaceType(ComInterfaceType.InterfaceIsIUnknown)]
interface IMMDevice {
    int Activate(ref Guid iid, int clsCtx, IntPtr p, [MarshalAs(UnmanagedType.IUnknown)] out object o);
    int OpenPropertyStore(int access, out IPropertyStore store);
    int GetId([MarshalAs(UnmanagedType.LPWStr)] out string id);
}
[Guid("886d8eeb-8cf2-4446-8d02-cdba1dbdcf99"), InterfaceType(ComInterfaceType.InterfaceIsIUnknown)]
interface IPropertyStore {
    int GetCount(out int c);
    int GetAt(int i, out PROPERTYKEY k);
    int GetValue(ref PROPERTYKEY k, out PROPVARIANT v);
}
[StructLayout(LayoutKind.Sequential)] struct PROPERTYKEY { public Guid fmtid; public int pid; }
[StructLayout(LayoutKind.Explicit)] struct PROPVARIANT { [FieldOffset(0)] public short vt; [FieldOffset(8)] public IntPtr p; [FieldOffset(16)] public IntPtr p2; }
[Guid("77AA99A0-1BD6-484F-8BC7-2C654C9A9B6F"), InterfaceType(ComInterfaceType.InterfaceIsIUnknown)]
interface IAudioSessionManager2 {
    int GetAudioSessionControl(IntPtr g, int f, out IntPtr c);
    int GetSimpleAudioVolume(IntPtr g, int f, out IntPtr v);
    int GetSessionEnumerator(out IAudioSessionEnumerator e);
}
[Guid("E2F5BB11-0570-40CA-ACDD-3AA01277DEE8"), InterfaceType(ComInterfaceType.InterfaceIsIUnknown)]
interface IAudioSessionEnumerator {
    int GetCount(out int c);
    int GetSession(int i, out IAudioSessionControl2 s);
}
[Guid("bfb7ff88-7239-4fc9-8fa2-07c950be9c6d"), InterfaceType(ComInterfaceType.InterfaceIsIUnknown)]
interface IAudioSessionControl2 {
    int GetState(out int state);
    int GetDisplayName([MarshalAs(UnmanagedType.LPWStr)] out string n);
    int SetDisplayName(string n, IntPtr ctx);
    int GetIconPath([MarshalAs(UnmanagedType.LPWStr)] out string p);
    int SetIconPath(string p, IntPtr ctx);
    int GetGroupingParam(out Guid g);
    int SetGroupingParam(ref Guid g, IntPtr ctx);
    int RegisterAudioSessionNotification(IntPtr n);
    int UnregisterAudioSessionNotification(IntPtr n);
    int GetSessionIdentifier([MarshalAs(UnmanagedType.LPWStr)] out string s);
    int GetSessionInstanceIdentifier([MarshalAs(UnmanagedType.LPWStr)] out string s);
    int GetProcessId(out uint pid);
}
[Guid("87CE5498-68D6-44E5-9215-6DA47EF883D8"), InterfaceType(ComInterfaceType.InterfaceIsIUnknown)]
interface ISimpleAudioVolume {
    int SetMasterVolume(float l, IntPtr ctx);
    int GetMasterVolume(out float l);
    int SetMute(bool m, IntPtr ctx);
    int GetMute(out bool m);
}
[Guid("C02216F6-8C67-4B5B-9D00-D008E73E0064"), InterfaceType(ComInterfaceType.InterfaceIsIUnknown)]
interface IAudioMeterInformation { int GetPeakValue(out float p); }
[Guid("5CDF2C82-841E-4546-9722-0CF74078229A"), InterfaceType(ComInterfaceType.InterfaceIsIUnknown)]
interface IAudioEndpointVolume {
    int RegisterControlChangeNotify(IntPtr n); int UnregisterControlChangeNotify(IntPtr n);
    int GetChannelCount(out int c); int SetMasterVolumeLevel(float l, IntPtr c); int SetMasterVolumeLevelScalar(float l, IntPtr c);
    int GetMasterVolumeLevel(out float l); int GetMasterVolumeLevelScalar(out float l);
    int SetChannelVolumeLevel(int ch, float l, IntPtr c); int SetChannelVolumeLevelScalar(int ch, float l, IntPtr c);
    int GetChannelVolumeLevel(int ch, out float l); int GetChannelVolumeLevelScalar(int ch, out float l);
    int SetMute(bool m, IntPtr c); int GetMute(out bool m);
}
[ComImport, Guid("BCDE0395-E52F-467C-8E3D-C4579291692E")] class MMDeviceEnumerator {}

public class SessionInfo {
    public string Device; public bool Default; public float DeviceVolume; public bool DeviceMute;
    public uint Pid; public int State; public float Volume; public bool Mute; public float Peak;
}

public static class AudioSessions {
    static string Name(IMMDevice d) {
        IPropertyStore s; d.OpenPropertyStore(0, out s);
        var k = new PROPERTYKEY { fmtid = new Guid("a45c254e-df1c-4efd-8020-67d146a850e0"), pid = 14 };
        PROPVARIANT v; s.GetValue(ref k, out v);
        return v.p == IntPtr.Zero ? "?" : Marshal.PtrToStringUni(v.p);
    }
    public static List<SessionInfo> List() { return Walk(null); }

    public static int Unmute(HashSet<uint> pids) { return Walk(pids).Count; }

    static List<SessionInfo> Walk(HashSet<uint> unmute) {
        var res = new List<SessionInfo>();
        var en = (IMMDeviceEnumerator)new MMDeviceEnumerator();
        IMMDevice def; en.GetDefaultAudioEndpoint(0, 1, out def);
        string defId; def.GetId(out defId);
        IMMDeviceCollection col; en.EnumAudioEndpoints(0, 1, out col);
        int n; col.GetCount(out n);
        for (int i = 0; i < n; i++) {
            IMMDevice d; col.Item(i, out d);
            string id; d.GetId(out id);
            object o;
            Guid ivol = typeof(IAudioEndpointVolume).GUID; d.Activate(ref ivol, 1, IntPtr.Zero, out o);
            var ev = (IAudioEndpointVolume)o; float dv; bool dm; ev.GetMasterVolumeLevelScalar(out dv); ev.GetMute(out dm);
            Guid iid = typeof(IAudioSessionManager2).GUID; d.Activate(ref iid, 1, IntPtr.Zero, out o);
            IAudioSessionEnumerator se; ((IAudioSessionManager2)o).GetSessionEnumerator(out se);
            int m; se.GetCount(out m);
            for (int j = 0; j < m; j++) {
                IAudioSessionControl2 c; se.GetSession(j, out c);
                var info = new SessionInfo { Device = Name(d), Default = id == defId, DeviceVolume = dv, DeviceMute = dm };
                c.GetProcessId(out info.Pid); c.GetState(out info.State);
                var sv = (ISimpleAudioVolume)c;
                if (unmute != null) {
                    if (unmute.Contains(info.Pid)) {
                        sv.SetMute(false, IntPtr.Zero); res.Add(info);
                    }
                    continue;
                }
                sv.GetMasterVolume(out info.Volume); sv.GetMute(out info.Mute);
                ((IAudioMeterInformation)c).GetPeakValue(out info.Peak);
                res.Add(info);
            }
        }
        return res;
    }
}
"@

$Unmute = @($Unmute | ForEach-Object { $_ -split ',' } | Where-Object { $_ })
if ($Unmute) {
    $deadline = (Get-Date).AddSeconds($Seconds)
    do {
        $pids = New-Object 'System.Collections.Generic.HashSet[uint32]'
        Get-Process -Name $Unmute -ErrorAction SilentlyContinue | ForEach-Object { [void]$pids.Add([uint32]$_.Id) }
        $n = [AudioSessions]::Unmute($pids)
        if ($n) { break }
        Start-Sleep -Milliseconds 200
    } while ((Get-Date) -lt $deadline)
    if ($n) { "Unmuted $n session(s) of: $($Unmute -join ', ')." }
    else { "No playing session of $($Unmute -join ', ') appeared within $Seconds s." }
    $Seconds = 0
}

$states = @{0 = 'inactive'; 1 = 'active'; 2 = 'expired'}
$end = (Get-Date).AddSeconds($Seconds)
$peaks = @{}
do {
    foreach ($s in [AudioSessions]::List()) {
        $key = "$($s.Device)|$($s.Pid)"
        if (-not $peaks.ContainsKey($key)) { $peaks[$key] = @{info = $s; max = 0.0} }
        $peaks[$key].info = $s
        $peaks[$key].max = [Math]::Max($peaks[$key].max, $s.Peak)
    }
    if ($Seconds -gt 0) { Start-Sleep -Milliseconds 50 }
} while ((Get-Date) -lt $end)

$peaks.Values | Where-Object { $_.info.State -ne 2 } | ForEach-Object {
    $s = $_.info
    $proc = if ($s.Pid) { (Get-Process -Id $s.Pid -ErrorAction SilentlyContinue).Name } else { 'system' }
    [pscustomobject]@{
        Device    = $s.Device + $(if ($s.Default) { ' [default]' } else { '' })
        DevVolume = '{0:P0}{1}' -f $s.DeviceVolume, $(if ($s.DeviceMute) { ' MUTED' } else { '' })
        Process   = "$proc ($($s.Pid))"
        State     = $states[$s.State]
        Volume    = '{0:P0}' -f $s.Volume
        Mute      = $s.Mute
        MaxPeak   = '{0:N3}' -f $_.max
    }
} | Sort-Object Device, Process | Format-Table -AutoSize
