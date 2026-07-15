;态极 Windows 安装脚本 (NSIS)
;================================
;
;使用方式：
;  1. 安装 NSIS: https://nsis.sourceforge.io/
;  2. 先运行 build_desktop.bat 打包
;  3. 右键此文件 → Compile NSIS Script
;
;输出：TaijiSetup.exe

!define APP_NAME "态极"
!define APP_VERSION "1.6.0"
!define APP_PUBLISHER "Taiji Project"
!define APP_EXE "Taiji.exe"
!define APP_DIR "Taiji"

; 安装器属性
Name "${APP_NAME} ${APP_VERSION}"
OutFile "TaijiSetup.exe"
InstallDir "$PROGRAMFILES\${APP_DIR}"
InstallDirRegKey HKLM "Software\${APP_DIR}" "InstallDir"
RequestExecutionLevel admin

; 界面
!include "MUI2.nsh"

!define MUI_ABORTWARNING
!define MUI_ICON "..\icon.ico"
!define MUI_UNICON "..\icon.ico"

!insertmacro MUI_PAGE_WELCOME
!insertmacro MUI_PAGE_LICENSE "..\LICENSE"
!insertmacro MUI_PAGE_DIRECTORY
!insertmacro MUI_PAGE_INSTFILES
!insertmacro MUI_PAGE_FINISH

!insertmacro MUI_UNPAGE_CONFIRM
!insertmacro MUI_UNPAGE_INSTFILES

!insertmacro MUI_LANGUAGE "SimpChinese"

; 安装区段
Section "安装"
    SetOutPath "$INSTDIR"

    ; 复制所有文件
    File /r "dist\Taiji\*.*"

    ; 创建卸载器
    WriteUninstaller "$INSTDIR\Uninstall.exe"

    ; 开始菜单快捷方式
    CreateDirectory "$SMPROGRAMS\${APP_NAME}"
    CreateShortCut "$SMPROGRAMS\${APP_NAME}\${APP_NAME}.lnk" "$INSTDIR\${APP_EXE}"
    CreateShortCut "$SMPROGRAMS\${APP_NAME}\卸载 ${APP_NAME}.lnk" "$INSTDIR\Uninstall.exe"

    ; 桌面快捷方式
    CreateShortCut "$DESKTOP\${APP_NAME}.lnk" "$INSTDIR\${APP_EXE}"

    ; 注册表
    WriteRegStr HKLM "Software\${APP_DIR}" "InstallDir" "$INSTDIR"
    WriteRegStr HKLM "Software\Microsoft\Windows\CurrentVersion\Uninstall\${APP_DIR}" \
        "DisplayName" "${APP_NAME} ${APP_VERSION}"
    WriteRegStr HKLM "Software\Microsoft\Windows\CurrentVersion\Uninstall\${APP_DIR}" \
        "UninstallString" "$INSTDIR\Uninstall.exe"
    WriteRegStr HKLM "Software\Microsoft\Windows\CurrentVersion\Uninstall\${APP_DIR}" \
        "Publisher" "${APP_PUBLISHER}"
    WriteRegDWORD HKLM "Software\Microsoft\Windows\CurrentVersion\Uninstall\${APP_DIR}" \
        "NoModify" 1
    WriteRegDWORD HKLM "Software\Microsoft\Windows\CurrentVersion\Uninstall\${APP_DIR}" \
        "NoRepair" 1

    ; 文件关联（可选）
    ; !define SHCNE_ASSOCCHANGED 0x08000000
    ; !define SHCNF_IDLIST 0
    ; System::Call 'shell32::SHChangeNotify(i ${SHCNE_ASSOCCHANGED}, i ${SHCNF_IDLIST}, i 0, i 0)'

SectionEnd

; 卸载区段
Section "Uninstall"
    ; 删除文件
    RMDir /r "$INSTDIR"

    ; 删除快捷方式
    RMDir /r "$SMPROGRAMS\${APP_NAME}"
    Delete "$DESKTOP\${APP_NAME}.lnk"

    ; 删除注册表
    DeleteRegKey HKLM "Software\${APP_DIR}"
    DeleteRegKey HKLM "Software\Microsoft\Windows\CurrentVersion\Uninstall\${APP_DIR}"
SectionEnd
