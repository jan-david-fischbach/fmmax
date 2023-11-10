"use strict";(self.webpackChunkdocs=self.webpackChunkdocs||[]).push([[7389],{3905:(e,t,n)=>{n.d(t,{Zo:()=>s,kt:()=>m});var r=n(7294);function a(e,t,n){return t in e?Object.defineProperty(e,t,{value:n,enumerable:!0,configurable:!0,writable:!0}):e[t]=n,e}function i(e,t){var n=Object.keys(e);if(Object.getOwnPropertySymbols){var r=Object.getOwnPropertySymbols(e);t&&(r=r.filter((function(t){return Object.getOwnPropertyDescriptor(e,t).enumerable}))),n.push.apply(n,r)}return n}function o(e){for(var t=1;t<arguments.length;t++){var n=null!=arguments[t]?arguments[t]:{};t%2?i(Object(n),!0).forEach((function(t){a(e,t,n[t])})):Object.getOwnPropertyDescriptors?Object.defineProperties(e,Object.getOwnPropertyDescriptors(n)):i(Object(n)).forEach((function(t){Object.defineProperty(e,t,Object.getOwnPropertyDescriptor(n,t))}))}return e}function l(e,t){if(null==e)return{};var n,r,a=function(e,t){if(null==e)return{};var n,r,a={},i=Object.keys(e);for(r=0;r<i.length;r++)n=i[r],t.indexOf(n)>=0||(a[n]=e[n]);return a}(e,t);if(Object.getOwnPropertySymbols){var i=Object.getOwnPropertySymbols(e);for(r=0;r<i.length;r++)n=i[r],t.indexOf(n)>=0||Object.prototype.propertyIsEnumerable.call(e,n)&&(a[n]=e[n])}return a}var d=r.createContext({}),u=function(e){var t=r.useContext(d),n=t;return e&&(n="function"==typeof e?e(t):o(o({},t),e)),n},s=function(e){var t=u(e.components);return r.createElement(d.Provider,{value:t},e.children)},p="mdxType",f={inlineCode:"code",wrapper:function(e){var t=e.children;return r.createElement(r.Fragment,{},t)}},c=r.forwardRef((function(e,t){var n=e.components,a=e.mdxType,i=e.originalType,d=e.parentName,s=l(e,["components","mdxType","originalType","parentName"]),p=u(n),c=a,m=p["".concat(d,".").concat(c)]||p[c]||f[c]||i;return n?r.createElement(m,o(o({ref:t},s),{},{components:n})):r.createElement(m,o({ref:t},s))}));function m(e,t){var n=arguments,a=t&&t.mdxType;if("string"==typeof e||a){var i=n.length,o=new Array(i);o[0]=c;var l={};for(var d in t)hasOwnProperty.call(t,d)&&(l[d]=t[d]);l.originalType=e,l[p]="string"==typeof e?e:a,o[1]=l;for(var u=2;u<i;u++)o[u]=n[u];return r.createElement.apply(null,o)}return r.createElement.apply(null,n)}c.displayName="MDXCreateElement"},8311:(e,t,n)=>{n.r(t),n.d(t,{assets:()=>d,contentTitle:()=>o,default:()=>f,frontMatter:()=>i,metadata:()=>l,toc:()=>u});var r=n(7462),a=(n(7294),n(3905));const i={id:"fields.amplitude_poynting_flux"},o=void 0,l={unversionedId:"API/fields.amplitude_poynting_flux",id:"API/fields.amplitude_poynting_flux",title:"fields.amplitude_poynting_flux",description:"fields.amplitudepoyntingflux",source:"@site/docs/API/fields.amplitude_poynting_flux.md",sourceDirName:"API",slug:"/API/fields.amplitude_poynting_flux",permalink:"/fmmax/API/fields.amplitude_poynting_flux",draft:!1,editUrl:"https://github.com/facebookresearch/fmmax/docs/API/fields.amplitude_poynting_flux.md",tags:[],version:"current",frontMatter:{id:"fields.amplitude_poynting_flux"},sidebar:"APISidebar",previous:{title:"fft.min_array_shape_for_expansion",permalink:"/fmmax/API/fft.min_array_shape_for_expansion"},next:{title:"fields.amplitudes_interior",permalink:"/fmmax/API/fields.amplitudes_interior"}},d={},u=[{value:"<code>fields.amplitude_poynting_flux</code>",id:"fieldsamplitude_poynting_flux",level:3},{value:"Args:",id:"args",level:4},{value:"Returns:",id:"returns",level:4}],s={toc:u},p="wrapper";function f(e){let{components:t,...n}=e;return(0,a.kt)(p,(0,r.Z)({},s,n,{components:t,mdxType:"MDXLayout"}),(0,a.kt)("h3",{id:"fieldsamplitude_poynting_flux"},(0,a.kt)("inlineCode",{parentName:"h3"},"fields.amplitude_poynting_flux")),(0,a.kt)("p",null,"Returns total Poynting flux for forward and backward eigenmodes."),(0,a.kt)("p",null,"This function decomposes the total field into components associated with\nthe forward and backward amplitudes, and returns the time-average flux in\neach order for these two components. The calculation follows section 5.1\nof ","[2012 Liu]","."),(0,a.kt)("p",null,"In the general case, a forward eigenmode may actually have negative\nPoynting flux, and therefore the quantities computed by this function\nshould not be interpreted as the total forward and backward flux, but only\nthe total flux associated with the forward and backward eigenmodes."),(0,a.kt)("p",null,"If the total forward and backward flux is desired, ",(0,a.kt)("inlineCode",{parentName:"p"},"directional_poynting_flux"),"\nshould be used instead. This function should only be used in the specific\ncase where the flux associated with the forward and backward eigenmodes is\nneeded."),(0,a.kt)("h4",{id:"args"},"Args:"),(0,a.kt)("ul",null,(0,a.kt)("li",{parentName:"ul"},(0,a.kt)("strong",{parentName:"li"},"forward_amplitude"),": The amplitude of the forward eigenmodes, with a\ntrailing batch dimension."),(0,a.kt)("li",{parentName:"ul"},(0,a.kt)("strong",{parentName:"li"},"backward_amplitude"),": The amplitude of the backward eigenmodes, at the\nsame location in space as the ",(0,a.kt)("inlineCode",{parentName:"li"},"forward_amplitude"),"."),(0,a.kt)("li",{parentName:"ul"},(0,a.kt)("strong",{parentName:"li"},"layer_solve_result"),": The results of the layer eigensolve.")),(0,a.kt)("h4",{id:"returns"},"Returns:"),(0,a.kt)("ul",null,(0,a.kt)("li",{parentName:"ul"},(0,a.kt)("strong",{parentName:"li"},"None"),": The Poynting flux associated with the forward and backward eigenmodes.")))}f.isMDXComponent=!0}}]);