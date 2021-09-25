<script lang="ts">
  import { Item, Text, PrimaryText, SecondaryText } from "@smui/list";
  export let bias: string = null;
</script>

<Item>
  <Text>
    <PrimaryText>Score</PrimaryText>
    <SecondaryText>
      <div class="main_result">{bias.toLowerCase()}</div>
    </SecondaryText>
  </Text>
</Item>

<style type="text/scss">
    
</style>
