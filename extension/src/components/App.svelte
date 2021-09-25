<script lang="ts">
  import type { Bias } from "../background";
  import type { Message } from "../content_script";

  import Data from "./Data.svelte";

  let biasScore: Bias = null;

  chrome.tabs.query(
    { active: true, lastFocusedWindow: true },
    (tabs: { url?: string }[]) => {
      var url = tabs[0].url;
      chrome.runtime.sendMessage({ type: "POPUP_INITIALIZED", url });
    }
  );

  chrome.runtime.onMessage.addListener((message: Message) => {
    if (message.type === "SEND_DATA") {
      biasScore = message.data;
    }
  });
</script>

<div class="container">
  {#if biasScore}
    <Data {biasScore} />
  {/if}
</div>

<style>
  .container {
    min-width: 250px;
  }
</style>
