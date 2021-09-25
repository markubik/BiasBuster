<script lang="ts">
  import List, { Item, Text, PrimaryText, SecondaryText } from "@smui/list";

  export let label: string = null;
  export let value: string = null;
</script>

<Item>
  <Text>
    <PrimaryText>{label}</PrimaryText>
    <SecondaryText class="single_result"
      ><div class="single_result">{value.toLowerCase()}</div></SecondaryText
    >
  </Text>
</Item>

<style type="text/scss">
  :global(.mdc-deprecated-list--two-line .mdc-deprecated-list-item) {
    height: 72px;
  }
  :global(.single_result) {
    text-transform: capitalize;
  }
  :global(.single_result::before) {
    content: "";
  }
</style>
