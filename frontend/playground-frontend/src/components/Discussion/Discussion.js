import React from "react";
import Giscus from "@giscus/react";

const Discussion = () => {
  return (
    <div className="sections" id="discussion">
      <Giscus
        repo="noah-iversen/dl-discussions"
        repoId="R_kgDOIpsdqg"
        category="General"
        categoryId="DIC_kwDOIpsdqs4CTLSI"
        mapping="url"
        strict="0"
        reactions-enabled="1"
        emit-metadata="0"
        input-position="top"
        theme="light"
        lang="en"
        loading="lazy"
      />
    </div>
  );
};
export default Discussion;
