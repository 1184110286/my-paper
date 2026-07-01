"""Constants extracted from the MalSnif paper."""

SELECTED_EVENTS = {
    # Process-related events
    "Process Start", "Thread Exit", "Process Exit", "Load Image", "Process Create", "Thread Create",
    # File-related events
    "LockFile", "ReadFile", "WriteFile", "SetSecurityFile", "SetValidDataLengthInformationFile",
    "SetDispositionInformationFile", "CloseFile", "CreateFile", "QueryFileInternalInformationFile",
    "CreateFileMapping", "UnlockFileSingle", "QueryEaInformationFile", "SetRenameInformationFile",
    # Registry-related events
    "RegSetInfoKey", "RegSetValue", "RegDeleteKey", "RegCreateKey", "RegUnloadKey", "RegDeleteValue",
    # Network-related events
    "TCP Reconnect", "TCP Connect", "UDP Send", "TCP Send", "TCP Receive", "TCP Disconnect",
    "TCP TCPCopy", "UDP Receive",
    # Query/setting-related events
    "NotifyChangeDirectory", "SetRenameInformationEx", "QueryStatInformation", "SetBasicInformationFile",
    "QuerySizeInformationVolume", "DeviceIoControl",
}

PROCESS_TYPES = {"process", "subject", "processobject", "thread", "PROCESS", "SUBJECT", "Process", "Subject"}

NODE_TYPE_NORMALIZATION = {
    "process": "PROCESS",
    "subject": "PROCESS",
    "thread": "PROCESS",
    "file": "FILE",
    "fileobject": "FILE",
    "artifact": "FILE",
    "registry": "REGISTRY",
    "registrykey": "REGISTRY",
    "netflow": "NETWORK",
    "netflowobject": "NETWORK",
    "socket": "NETWORK",
    "network": "NETWORK",
    "event": "EVENT",
    "unknown": "UNKNOWN",
}

NODE_TYPE_TO_ID = {
    "PROCESS": 0,
    "FILE": 1,
    "REGISTRY": 2,
    "NETWORK": 3,
    "EVENT": 4,
    "UNKNOWN": 5,
}
