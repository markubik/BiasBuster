<script lang="ts">
  import { Item, Text, PrimaryText, SecondaryText } from "@smui/list";
  import Tooltip, { Wrapper } from "@smui/tooltip";

  export let bias: string = null;
</script>

<Item>
  <Text>
    <Wrapper>
      <PrimaryText>
        Bias score <i class="fa fa-question" style="font-size:16px;" />
      </PrimaryText>
      <Tooltip>
        A data-driven metric meant to approximate the bias of the article on the
        website you are currently viewing. Calculated based on metrics listed
        below: hate speech, political charge and establishment stance. Possible
        values are: balanced, taking a stance, misleading, biased or strongly
        biased.
      </Tooltip>
    </Wrapper>
    <SecondaryText>
      <div class="main_result">{bias}</div>
    </SecondaryText>
  </Text>
</Item>

<style type="text/scss">
</style>
