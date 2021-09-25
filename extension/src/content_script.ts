import type { Bias } from "./background";

export type MessageType = 'PAGE_INITIALIZED' | 'POPUP_INITIALIZED' | 'SEND_DATA' | 'ERROR';

export type PageInitializedMessage = {
    type: 'PAGE_INITIALIZED'
}

export type PopupInitializedMessage = {
    type: 'POPUP_INITIALIZED',
    url: string
}

export type SendDataMessage = {
    type: 'SEND_DATA',
    data: Bias
}

export type ErrorMessage = {
    type: 'ERROR',
    data: string
}

export type Message = PageInitializedMessage | PopupInitializedMessage | SendDataMessage | ErrorMessage;
chrome.runtime.sendMessage({ type: "PAGE_INITIALIZED" });