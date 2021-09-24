<script lang="ts">
  import type { Message } from "../content_script";

  import Reader from "./Reader.svelte";

  let articleTitle: string = null;

  chrome.tabs.query(
    { active: true, lastFocusedWindow: true },
    (tabs: { url?: string }[]) => {
      var url = tabs[0].url;
      chrome.runtime.sendMessage({ type: "POPUP_INITIALIZED", url });
    }
  );

  chrome.runtime.onMessage.addListener((message: Message) => {
    if (message.type === "SEND_DATA") {
      //TODO Present results
    }
  });
</script>

<div class="container">
  <div>
    {#if articleTitle}<span class="success">Article title: {articleTitle}</span>
    {/if}
  </div>
  <Reader />
</div>

<style>
  .container {
    min-width: 250px;
  }

  .success {
    color: #2ecc71;
    font-weight: bold;
  }
</style>
