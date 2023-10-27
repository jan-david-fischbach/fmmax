"use strict";(self.webpackChunkdocs=self.webpackChunkdocs||[]).push([[3505],{3905:(e,r,t)=>{t.d(r,{Zo:()=>c,kt:()=>f});var n=t(7294);function o(e,r,t){return r in e?Object.defineProperty(e,r,{value:t,enumerable:!0,configurable:!0,writable:!0}):e[r]=t,e}function a(e,r){var t=Object.keys(e);if(Object.getOwnPropertySymbols){var n=Object.getOwnPropertySymbols(e);r&&(n=n.filter((function(r){return Object.getOwnPropertyDescriptor(e,r).enumerable}))),t.push.apply(t,n)}return t}function i(e){for(var r=1;r<arguments.length;r++){var t=null!=arguments[r]?arguments[r]:{};r%2?a(Object(t),!0).forEach((function(r){o(e,r,t[r])})):Object.getOwnPropertyDescriptors?Object.defineProperties(e,Object.getOwnPropertyDescriptors(t)):a(Object(t)).forEach((function(r){Object.defineProperty(e,r,Object.getOwnPropertyDescriptor(t,r))}))}return e}function s(e,r){if(null==e)return{};var t,n,o=function(e,r){if(null==e)return{};var t,n,o={},a=Object.keys(e);for(n=0;n<a.length;n++)t=a[n],r.indexOf(t)>=0||(o[t]=e[t]);return o}(e,r);if(Object.getOwnPropertySymbols){var a=Object.getOwnPropertySymbols(e);for(n=0;n<a.length;n++)t=a[n],r.indexOf(t)>=0||Object.prototype.propertyIsEnumerable.call(e,t)&&(o[t]=e[t])}return o}var l=n.createContext({}),u=function(e){var r=n.useContext(l),t=r;return e&&(t="function"==typeof e?e(r):i(i({},r),e)),t},c=function(e){var r=u(e.components);return n.createElement(l.Provider,{value:r},e.children)},p="mdxType",m={inlineCode:"code",wrapper:function(e){var r=e.children;return n.createElement(n.Fragment,{},r)}},d=n.forwardRef((function(e,r){var t=e.components,o=e.mdxType,a=e.originalType,l=e.parentName,c=s(e,["components","mdxType","originalType","parentName"]),p=u(t),d=o,f=p["".concat(l,".").concat(d)]||p[d]||m[d]||a;return t?n.createElement(f,i(i({ref:r},c),{},{components:t})):n.createElement(f,i({ref:r},c))}));function f(e,r){var t=arguments,o=r&&r.mdxType;if("string"==typeof e||o){var a=t.length,i=new Array(a);i[0]=d;var s={};for(var l in r)hasOwnProperty.call(r,l)&&(s[l]=r[l]);s.originalType=e,s[p]="string"==typeof e?e:o,i[1]=s;for(var u=2;u<a;u++)i[u]=t[u];return n.createElement.apply(null,i)}return n.createElement.apply(null,t)}d.displayName="MDXCreateElement"},708:(e,r,t)=>{t.r(r),t.d(r,{assets:()=>l,contentTitle:()=>i,default:()=>m,frontMatter:()=>a,metadata:()=>s,toc:()=>u});var n=t(7462),o=(t(7294),t(3905));const a={id:"sources.polarization_terms"},i=void 0,s={unversionedId:"API/sources.polarization_terms",id:"API/sources.polarization_terms",title:"sources.polarization_terms",description:"sources.polarization_terms",source:"@site/docs/API/sources.polarization_terms.md",sourceDirName:"API",slug:"/API/sources.polarization_terms",permalink:"/fmmax/API/sources.polarization_terms",draft:!1,editUrl:"https://github.com/facebookresearch/fmmax/docs/API/sources.polarization_terms.md",tags:[],version:"current",frontMatter:{id:"sources.polarization_terms"},sidebar:"APISidebar",previous:{title:"sources.gaussian_source",permalink:"/fmmax/API/sources.gaussian_source"},next:{title:"utils.absolute_axes",permalink:"/fmmax/API/utils.absolute_axes"}},l={},u=[{value:"<code>sources.polarization_terms</code>",id:"sourcespolarization_terms",level:3},{value:"Args:",id:"args",level:4},{value:"Returns:",id:"returns",level:4}],c={toc:u},p="wrapper";function m(e){let{components:r,...t}=e;return(0,o.kt)(p,(0,n.Z)({},c,t,{components:r,mdxType:"MDXLayout"}),(0,o.kt)("h3",{id:"sourcespolarization_terms"},(0,o.kt)("inlineCode",{parentName:"h3"},"sources.polarization_terms")),(0,o.kt)("p",null,"Computes the polarization terms for currents on the real-space grid."),(0,o.kt)("p",null,"The polarization terms are discussed in section 7 of ","[1999 Whittaker]","."),(0,o.kt)("h4",{id:"args"},"Args:"),(0,o.kt)("ul",null,(0,o.kt)("li",{parentName:"ul"},(0,o.kt)("strong",{parentName:"li"},"jx"),": The Fourier-transformed x-oriented dipole amplitude, with a\ntrailing batch dimension."),(0,o.kt)("li",{parentName:"ul"},(0,o.kt)("strong",{parentName:"li"},"jy"),": The Fourier-transformed y-oriented dipole amplitude."),(0,o.kt)("li",{parentName:"ul"},(0,o.kt)("strong",{parentName:"li"},"jz"),": The Fourier-transformed z-oriented dipole amplitude."),(0,o.kt)("li",{parentName:"ul"},(0,o.kt)("strong",{parentName:"li"},"layer_solve_result"),": The results of the layer eigensolve.")),(0,o.kt)("h4",{id:"returns"},"Returns:"),(0,o.kt)("ul",null,(0,o.kt)("li",{parentName:"ul"},(0,o.kt)("strong",{parentName:"li"},"None"),": The polarization vector containing parallel and z-oriented terms.")))}m.isMDXComponent=!0}}]);