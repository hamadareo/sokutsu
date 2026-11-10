/* 配る版(他人の Mac 向け Sokutsu.app)だけで読む(scripts/build_dist.sh が index.html に足す)。
   処理場所の札: 真ん中は「自分のサーバー」(config.json の server_url。空なら札を隠す)、
   「このMac」は同梱の翻訳エンジン(llama-server + Gemma 3 4B)。ユーザー本人の版では読まない。 */
(function(){
  const T=TSUYAKU;
  Object.assign(I18N.en,{
    "自分のサーバー":"Your server",
    "自分のサーバーがあればそれで、無ければこの Mac で翻訳します":"Uses your own server when one is set, otherwise translates on this Mac",
    "設定した自分の翻訳サーバー(OpenAI 互換)で翻訳します":"Translates on the server you set (OpenAI-compatible)",
    "この Mac の翻訳エンジン({m})で翻訳します — ネットワークは使いません":"Translates with the engine on this Mac ({m}) — no network needed",
    "自分のサーバーに接続できました —{m}で翻訳します":"Connected to your server —{m}will translate",
    "自分のサーバーに届きません — config.json の server_url を確かめてください":"Can't reach your server — check server_url in config.json",
    "自分のサーバーに届かないので、この Mac の翻訳エンジンで翻訳します":"Your server is not reachable — translating with the engine on this Mac"});
  Object.assign(I18N.zh,{
    "自分のサーバー":"自己的服务器",
    "自分のサーバーがあればそれで、無ければこの Mac で翻訳します":"设置了自己的服务器就用它，否则在这台 Mac 上翻译",
    "設定した自分の翻訳サーバー(OpenAI 互換)で翻訳します":"用您设置的翻译服务器（兼容 OpenAI）翻译",
    "この Mac の翻訳エンジン({m})で翻訳します — ネットワークは使いません":"用这台 Mac 上的翻译引擎（{m}）翻译 — 不使用网络",
    "自分のサーバーに接続できました —{m}で翻訳します":"已连接自己的服务器 —用{m}翻译",
    "自分のサーバーに届きません — config.json の server_url を確かめてください":"无法连接自己的服务器 — 请检查 config.json 中的 server_url",
    "自分のサーバーに届かないので、この Mac の翻訳エンジンで翻訳します":"无法连接自己的服务器，改用这台 Mac 上的翻译引擎"});
  Object.assign(I18N.ko,{
    "自分のサーバー":"내 서버",
    "自分のサーバーがあればそれで、無ければこの Mac で翻訳します":"내 서버가 설정되어 있으면 그것으로, 없으면 이 Mac에서 번역합니다",
    "設定した自分の翻訳サーバー(OpenAI 互換)で翻訳します":"설정한 내 번역 서버(OpenAI 호환)로 번역합니다",
    "この Mac の翻訳エンジン({m})で翻訳します — ネットワークは使いません":"이 Mac의 번역 엔진({m})으로 번역합니다 — 네트워크를 쓰지 않습니다",
    "自分のサーバーに接続できました —{m}で翻訳します":"내 서버에 연결했습니다 —{m}로 번역합니다",
    "自分のサーバーに届きません — config.json の server_url を確かめてください":"내 서버에 닿지 않습니다 — config.json의 server_url을 확인하세요",
    "自分のサーバーに届かないので、この Mac の翻訳エンジンで翻訳します":"내 서버에 닿지 않아 이 Mac의 번역 엔진으로 번역합니다"});

  const chips=()=>document.querySelectorAll("#engines .pair");
  const SRV=(chips()[1]||{dataset:{}}).dataset.engine;   // 真ん中の札の値(自分のサーバー)
  T.ENG_NOTE={auto:"自分のサーバーがあればそれで、無ければこの Mac で翻訳します",
    mac:"この Mac だけで翻訳します。ネットワーク不要(長い文は少し遅くなることがあります)"};
  T.ENG_NOTE[SRV]="設定した自分の翻訳サーバー(OpenAI 互換)で翻訳します";

  // 自分のサーバーが無ければ札を隠す(2つにする)
  function serverChip(on){
    const c=chips()[1], box=document.querySelector("#engines"); if(!c||!box)return;
    c._tja="自分のサーバー"; c.setAttribute("data-t",""); c.innerHTML=T.L("自分のサーバー");
    c.hidden=!on; box.style.gridTemplateColumns=on?"":"repeat(2,1fr)";
  }
  T.engineShow=function(e){
    this._eng=e;
    const h=this.$("#startHint"); if(!h) return;
    const p=(e&&e.reachable)||{}, Lb=(e&&e.labels)||{}, M=(e&&e.models)||{};
    const has=k=>p[Lb[k]]!=null, ok=k=>p[Lb[k]]===true;
    const m=k=>({m:M[Lb[k]]?(" "+M[Lb[k]]+" "):" "});
    const eng=(e&&e.engine)||"auto";
    serverChip(has(SRV));
    let t;
    const local=this.L("この Mac の翻訳エンジン({m})で翻訳します — ネットワークは使いません",{m:M[Lb.mac]||"Gemma 3 4B"});
    if(eng==="mac"||!has(SRV)) t=local;
    else if(eng===SRV) t=ok(SRV)?this.L("自分のサーバーに接続できました —{m}で翻訳します",m(SRV))
      :this.L("自分のサーバーに届きません — config.json の server_url を確かめてください");
    else t=ok(SRV)?this.L("自分のサーバーに接続できました —{m}で翻訳します",m(SRV))
      :this.L("自分のサーバーに届かないので、この Mac の翻訳エンジンで翻訳します");
    h.textContent=t;
  };
  serverChip(false);
  const h=document.querySelector("#startHint");
  if(h)h.textContent=T.L("この Mac の翻訳エンジン({m})で翻訳します — ネットワークは使いません",{m:"Gemma 3 4B"});
  T.segNotes();
})();
